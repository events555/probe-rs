//! ChibiOS/RT thread awareness.
//!
//! ChibiOS embeds a `chdebug_t` structure in the binary (via `chregistry.h`)
//! that contains all the byte-offsets into the thread struct and the system
//! struct. This makes ChibiOS uniquely easy to support — no hardcoded offsets,
//! no version-specific tables. We just read the signature and follow where it
//! points.
//!
//! # Detection
//!
//! We look for the `ch_debug` symbol. The first 5 bytes must be `"main\0"`.
//! We also need `ch_system` to locate OS instances.
//!
//! # Thread enumeration
//!
//! The `chdebug_t` signature describes how many OS instances exist and where
//! their thread registries live (as byte offsets within the system and instance
//! structs). We walk the registry linked list for each instance.
//!
//! # Register recovery
//!
//! Each thread's `port_context.sp` points to a `port_intctx` on its stack.
//! Without FPU: `{r4-r11, lr}` (9 words, 0x24 bytes).
//! With FPU:    `{s16-s31, r4-r11, lr}` (25 words, 0x64 bytes).
//!
//! If the saved LR is an EXC_RETURN value (bits [31:4] = 0xFFFFFFF), the
//! thread was preempted by an interrupt and a hardware exception frame sits
//! above the software frame. We parse that frame to recover the real PC,
//! caller-saved registers, and reconstruct the correct SP.

use super::{read_cstring, ResolvedSymbols, RtosAwareness, RtosThread, SavedRegister, SymbolQuery};
use crate::{Core, Error, RegisterId};
use crate::MemoryInterface;

/// CPACR register address (Coprocessor Access Control Register).
const CPACR_ADDR: u64 = 0xE000_ED88;

/// Minimum size of the `chdebug_t` signature we need to read.
/// Covers through `off_inst_rfcu` (byte 42).
const CHDEBUG_MIN_SIZE: usize = 43;

/// Thread state names, indexed by the `state` byte in the thread struct.
/// This table matches ChibiOS/RT 3.x through 6.x. Older (2.x) or future
/// versions may use different indices; out-of-range values display as "state=N".
const THREAD_STATES: &[&str] = &[
    "Ready",
    "Current",
    "WtStart",
    "Suspended",
    "Queued",
    "WtSem",
    "WtMtx",
    "WtCond",
    "Sleeping",
    "WtExit",
    "WtOrEvt",
    "WtAndEvt",
    "SndMsgQ",
    "SndMsg",
    "WtMsg",
    "Final",
];

/// Maximum thread name length we'll read from target memory.
const MAX_THREAD_NAME: usize = 64;

/// Maximum number of threads before we assume corruption.
const MAX_THREADS: usize = 256;

/// The ChibiOS memory signature (`chdebug_t`), read from the `ch_debug` symbol.
///
/// All values are byte offsets unless noted otherwise. The layout matches
/// `chregistry.h` in ChibiOS/RT.
#[derive(Debug)]
#[allow(dead_code)]
struct ChibiDebugSignature {
    // -- Basic thread struct offsets (always present) --
    ptr_size: u8,
    time_size: u8,
    thread_size: u8,
    off_prio: u8,
    off_ctx: u8,
    off_newer: u8,
    off_older: u8,
    off_name: u8,
    off_stklimit: u8,
    off_state: u8,
    off_flags: u8,
    off_refs: u8,
    off_preempt: u8,
    off_time: u8,
    intctx_size: u8,
    interval_size: u8,

    // -- Instance-aware fields (ChibiOS 21.11+) --
    instances_num: u8,
    off_sys_state: u8,
    off_sys_instances: u8,
    off_sys_reglist: u8,
    off_inst_rlist_current: u8,
    off_inst_rlist: u8,
    off_inst_vtlist: u8,
    off_inst_reglist: u8,
    off_inst_core_id: u8,

    /// True if the signature is large enough to contain instance-aware fields.
    has_instance_info: bool,
}

impl ChibiDebugSignature {
    /// Read and validate the signature from target memory.
    fn read_from(core: &mut Core, addr: u64) -> Result<Self, Error> {
        // First read the size byte to know how much to read.
        let mut header = [0u8; 6];
        core.read_8(addr, &mut header)?;

        // Bytes 0..5: identifier, must be "main\0"
        if &header[0..5] != b"main\0" {
            return Err(Error::Other(
                "ChibiOS debug signature not found (expected 'main\\0' magic)".to_string(),
            ));
        }

        let struct_size = header[5] as usize;
        if struct_size < 20 {
            return Err(Error::Other(format!(
                "ChibiOS debug signature too small: {struct_size} < 20"
            )));
        }

        // Read the full signature.
        let read_size = struct_size.min(CHDEBUG_MIN_SIZE);
        let mut buf = vec![0u8; read_size];
        core.read_8(addr, &mut buf)?;

        // Byte 8: pointer size, must be 4 for 32-bit targets.
        if buf[8] != 4 {
            return Err(Error::Other(format!(
                "ChibiOS reports pointer size {}, expected 4",
                buf[8]
            )));
        }

        let has_instance_info = struct_size >= CHDEBUG_MIN_SIZE;

        Ok(Self {
            ptr_size: buf[8],
            time_size: buf[9],
            thread_size: buf[10],
            off_prio: buf[11],
            off_ctx: buf[12],
            off_newer: buf[13],
            off_older: buf[14],
            off_name: buf[15],
            off_stklimit: buf[16],
            off_state: buf[17],
            off_flags: buf[18],
            off_refs: buf[19],
            off_preempt: if read_size > 20 { buf[20] } else { 0 },
            off_time: if read_size > 21 { buf[21] } else { 0 },
            // bytes 22-25: reserved
            intctx_size: if read_size > 26 { buf[26] } else { 0 },
            interval_size: if read_size > 27 { buf[27] } else { 0 },
            instances_num: if read_size > 28 { buf[28] } else { 1 },
            off_sys_state: if read_size > 29 { buf[29] } else { 0 },
            off_sys_instances: if read_size > 30 { buf[30] } else { 0 },
            off_sys_reglist: if read_size > 31 { buf[31] } else { 0 },
            // bytes 33-36: off_sys_reserved
            off_inst_rlist_current: if read_size > 37 { buf[37] } else { 0 },
            off_inst_rlist: if read_size > 38 { buf[38] } else { 0 },
            off_inst_vtlist: if read_size > 39 { buf[39] } else { 0 },
            off_inst_reglist: if read_size > 40 { buf[40] } else { 0 },
            off_inst_core_id: if read_size > 41 { buf[41] } else { 0 },
            has_instance_info,
        })
    }
}

/// Software context frame layout, which determines how to find callee-saved
/// registers on the thread's stack.
#[derive(Debug, Clone, Copy)]
struct SwFrameLayout {
    /// Total size in bytes of the software-saved context frame (`port_intctx`).
    size: u64,
    /// Byte offset from `sp` to the first general-purpose register (r4).
    gpr_offset: u64,
}

/// Known frame layouts for Cortex-M.
const SW_FRAME_NO_FPU: SwFrameLayout = SwFrameLayout {
    size: 0x24,       // r4-r11, lr = 9 * 4
    gpr_offset: 0x00,
};

const SW_FRAME_WITH_FPU: SwFrameLayout = SwFrameLayout {
    size: 0x64,       // s16-s31 (16) + r4-r11, lr (9) = 25 * 4
    gpr_offset: 0x40, // skip 16 FPU regs
};

impl SwFrameLayout {
    /// Determine the software frame layout.
    ///
    /// Prefers the `intctx_size` field from the ChibiOS debug signature (which
    /// reflects the firmware's compile-time FPU configuration) over reading the
    /// hardware CPACR register (which only tells us if the hardware *has* an FPU,
    /// not whether the firmware uses it).
    fn detect(core: &mut Core, sig: &ChibiDebugSignature) -> Result<Self, Error> {
        if sig.intctx_size > 0 {
            // The signature tells us the exact size of port_intctx.
            let size = sig.intctx_size as u64;
            let gpr_offset = if size > SW_FRAME_NO_FPU.size {
                // FPU regs are at the start; GPRs follow.
                size - SW_FRAME_NO_FPU.size
            } else {
                0
            };
            return Ok(Self { size, gpr_offset });
        }

        // Fallback for older signatures without intctx_size: read CPACR.
        let cpacr = core.read_word_32(CPACR_ADDR)?;
        if cpacr & 0x00F0_0000 != 0 {
            Ok(SW_FRAME_WITH_FPU)
        } else {
            Ok(SW_FRAME_NO_FPU)
        }
    }
}

/// Per-instance registry info derived from `ch_system`.
#[derive(Debug)]
struct InstanceInfo {
    /// Address of the instance struct in target memory.
    instance_addr: u64,
    /// Address of the registry list root for this instance.
    reglist_addr: u64,
    /// Address of the `rlist.current` pointer for this instance.
    current_thread_ptr_addr: u64,
}

/// ChibiOS/RT thread awareness backend.
pub struct ChibiOsAwareness {
    sig: ChibiDebugSignature,
    instances: Vec<InstanceInfo>,
    sw_frame: SwFrameLayout,
    /// Thread ID of the currently running thread (updated on each `threads()` call).
    /// Used to reject `thread_registers()` for the running thread, which has no
    /// valid saved context.
    current_thread_id: Option<u64>,
}

impl ChibiOsAwareness {
    /// Try to detect ChibiOS on the target.
    pub fn detect(core: &mut Core, symbols: &ResolvedSymbols) -> Option<Box<dyn RtosAwareness>> {
        let ch_debug_addr = *symbols.get("ch_debug")?;

        let sig = match ChibiDebugSignature::read_from(core, ch_debug_addr) {
            Ok(s) => s,
            Err(e) => {
                tracing::debug!("ChibiOS signature read failed: {e}");
                return None;
            }
        };

        tracing::debug!("ChibiOS signature: {sig:?}");

        let instances = if sig.has_instance_info {
            // Modern ChibiOS: derive instance addresses from ch_system + offsets.
            let ch_system_addr = match symbols.get("ch_system") {
                Some(&addr) => addr,
                None => {
                    tracing::debug!(
                        "ChibiOS: ch_debug has instance info but 'ch_system' symbol not found"
                    );
                    return None;
                }
            };

            let num = sig.instances_num.max(1) as usize;
            let mut instances = Vec::with_capacity(num);

            for i in 0..num {
                // Each instance is ptr_size apart in the instances array.
                let instance_ptr_addr = ch_system_addr
                    + sig.off_sys_instances as u64
                    + (i as u64) * (sig.ptr_size as u64);

                let instance_addr = match core.read_word_32(instance_ptr_addr) {
                    Ok(addr) => addr as u64,
                    Err(e) => {
                        tracing::debug!(
                            "ChibiOS: failed to read instance {i} pointer at {instance_ptr_addr:#x}: {e}"
                        );
                        continue;
                    }
                };

                if instance_addr == 0 {
                    tracing::debug!("ChibiOS: instance {i} pointer is NULL, skipping");
                    continue;
                }

                let reglist_addr = instance_addr + sig.off_inst_reglist as u64;
                let current_thread_ptr_addr =
                    instance_addr + sig.off_inst_rlist_current as u64;

                tracing::debug!(
                    "ChibiOS instance {i}: addr={instance_addr:#x}, \
                     reglist={reglist_addr:#x}, current_ptr={current_thread_ptr_addr:#x}"
                );

                instances.push(InstanceInfo {
                    instance_addr,
                    reglist_addr,
                    current_thread_ptr_addr,
                });
            }

            if instances.is_empty() {
                tracing::debug!("ChibiOS: no valid instances found");
                return None;
            }

            instances
        } else {
            // Legacy ChibiOS (pre-instance): fall back to `rlist` or `ch` symbols.
            let rlist_addr = symbols
                .get("rlist")
                .or_else(|| symbols.get("ch"))
                .copied();

            match rlist_addr {
                Some(addr) => {
                    tracing::debug!(
                        "ChibiOS legacy mode: rlist at {addr:#x}"
                    );
                    vec![InstanceInfo {
                        instance_addr: addr,
                        reglist_addr: addr,
                        // Legacy: current thread pointer is at rlist + off_name
                        // (the name offset in the ready list header coincides with current).
                        current_thread_ptr_addr: addr + sig.off_name as u64,
                    }]
                }
                None => {
                    tracing::debug!(
                        "ChibiOS: legacy signature but no 'rlist'/'ch' symbol found"
                    );
                    return None;
                }
            }
        };

        let sw_frame = SwFrameLayout::detect(core, &sig).ok()?;

        Some(Box::new(Self {
            sig,
            instances,
            sw_frame,
            current_thread_id: None,
        }))
    }
}

impl RtosAwareness for ChibiOsAwareness {
    fn required_symbols(&self) -> Vec<SymbolQuery> {
        vec![
            SymbolQuery {
                name: "ch_debug",
                optional: false,
            },
            SymbolQuery {
                name: "ch_system",
                optional: true,
            },
            // Legacy fallbacks (ChibiOS 2.x / early 3.x without instance support).
            SymbolQuery {
                name: "ch",
                optional: true,
            },
            SymbolQuery {
                name: "rlist",
                optional: true,
            },
        ]
    }

    fn threads(&mut self, core: &mut Core) -> Result<Vec<RtosThread>, Error> {
        let sig = &self.sig;
        let mut all_threads = Vec::new();
        let mut found_current: Option<u64> = None;

        for inst in &self.instances {
            let reglist = inst.reglist_addr;

            // Read the current thread pointer for this instance.
            let current_thread = core.read_word_32(inst.current_thread_ptr_addr)? as u64;

            // Walk the registry doubly-linked list.
            let mut count = 0usize;
            let mut addr = core.read_word_32(reglist)? as u64;

            // Validate list integrity in first pass.
            let mut prev = reglist;
            let _start_addr = addr;
            loop {
                if addr == 0 {
                    return Err(Error::Other(
                        "ChibiOS registry: NULL pointer in thread list".to_string(),
                    ));
                }
                if addr == reglist {
                    break; // Full loop completed
                }

                // Integrity: check backward pointer matches where we came from.
                // The queue node's backward pointer is at (off_older - off_newer) from the
                // forward pointer, since the node is embedded at off_newer within thread_t.
                let back_ptr_offset = (sig.off_older - sig.off_newer) as u64;
                let older = core.read_word_32(addr + back_ptr_offset)? as u64;
                if older != prev {
                    tracing::warn!(
                        "ChibiOS registry: backward pointer mismatch \
                         (older={older:#x}, expected prev={prev:#x}), continuing anyway"
                    );
                }

                count += 1;
                if count > MAX_THREADS {
                    return Err(Error::Other(format!(
                        "ChibiOS registry: more than {MAX_THREADS} threads, likely corrupted"
                    )));
                }

                prev = addr;
                addr = core.read_word_32(addr)? as u64; // first pointer = next
            }

            // Second pass: collect thread details.
            // The registry queue nodes are embedded in thread_t at off_newer/off_older.
            // To get the thread_t base address from a queue node, subtract the offset.
            addr = core.read_word_32(reglist)? as u64;

            while addr != reglist {
                // The queue node is at offset off_newer within thread_t.
                // So thread_base = queue_node_addr - off_newer.
                let thread_base = addr - sig.off_newer as u64;

                let name_ptr = core.read_word_32(thread_base + sig.off_name as u64)? as u64;
                let name = if name_ptr != 0 {
                    read_cstring(core, name_ptr, MAX_THREAD_NAME)?
                } else {
                    String::new()
                };

                let state_byte = core.read_word_8(thread_base + sig.off_state as u64)?;
                let state = THREAD_STATES
                    .get(state_byte as usize)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("state={state_byte}"));

                // ChibiOS tprio_t is typically u32. Read full word for endianness
                // correctness, then truncate to u8 for display.
                let priority = core.read_word_32(thread_base + sig.off_prio as u64)? as u8;

                let is_current = thread_base == current_thread;
                if is_current {
                    found_current = Some(thread_base);
                }

                all_threads.push(RtosThread {
                    id: thread_base,
                    name: if name.is_empty() {
                        "unnamed".into()
                    } else {
                        name
                    },
                    state,
                    priority,
                    is_current,
                });

                addr = core.read_word_32(addr)? as u64; // next in queue
            }
        }

        self.current_thread_id = found_current;
        Ok(all_threads)
    }

    fn thread_registers(
        &self,
        core: &mut Core,
        thread_id: u64,
    ) -> Result<Vec<SavedRegister>, Error> {
        // Guard: the current thread's saved context is stale — the caller
        // should read registers directly from the core instead.
        if self.current_thread_id == Some(thread_id) {
            return Err(Error::Other(
                "Cannot read saved registers for the currently running thread; \
                 use live core registers instead"
                    .to_string(),
            ));
        }

        // Read the saved stack pointer from the thread's port_context.
        let sp = core.read_word_32(thread_id + self.sig.off_ctx as u64)? as u64;

        let gpr_base = sp + self.sw_frame.gpr_offset;

        // Callee-saved registers from the software context frame.
        const SW_REGS: &[(u16, u64)] = &[
            (4, 0x00),  // r4
            (5, 0x04),  // r5
            (6, 0x08),  // r6
            (7, 0x0C),  // r7
            (8, 0x10),  // r8
            (9, 0x14),  // r9
            (10, 0x18), // r10
            (11, 0x1C), // r11
        ];

        let mut regs = Vec::with_capacity(17);

        for &(reg_num, offset) in SW_REGS {
            let val = core.read_word_32(gpr_base + offset)?;
            regs.push(SavedRegister {
                id: RegisterId(reg_num),
                value: Some(val),
            });
        }

        // The saved LR from the context switch.
        let saved_lr = core.read_word_32(gpr_base + 0x20)?;

        // Check if the saved LR is an EXC_RETURN value (Cortex-M exception
        // return magic). This means the thread was preempted by an interrupt,
        // and the hardware exception frame sits above the software frame on
        // the stack. EXC_RETURN values have bits [31:4] set to 0xFFFFFFF.
        let is_exc_return = saved_lr & 0xFFFF_FFF0 == 0xFFFF_FFF0;

        if is_exc_return {
            // The hardware exception frame is above the software frame:
            //   [sw_frame | hw_exception_frame | ... rest of stack]
            //   ^sp       ^hw_base
            //
            // The hw frame layout (without FPU stacking):
            //   +0x00: R0
            //   +0x04: R1
            //   +0x08: R2
            //   +0x0C: R3
            //   +0x10: R12
            //   +0x14: LR (the real return address)
            //   +0x18: PC (the real PC where the thread was interrupted)
            //   +0x1C: xPSR
            //   (if FPU stacking: +0x20..+0x60: S0-S15, FPSCR, reserved)
            //
            // EXC_RETURN bit 4: 0 = FPU context stacked, 1 = no FPU stacking.
            let hw_fpu_stacked = saved_lr & (1 << 4) == 0;
            let hw_frame_size: u64 = if hw_fpu_stacked {
                0x68 // 26 words: 8 standard + 18 FPU (S0-S15, FPSCR, reserved)
            } else {
                0x20 // 8 words: R0, R1, R2, R3, R12, LR, PC, xPSR
            };

            let hw_base = sp + self.sw_frame.size;

            // Read the real register values from the hardware exception frame.
            let hw_r0 = core.read_word_32(hw_base)?;
            let hw_r1 = core.read_word_32(hw_base + 0x04)?;
            let hw_r2 = core.read_word_32(hw_base + 0x08)?;
            let hw_r3 = core.read_word_32(hw_base + 0x0C)?;
            let hw_r12 = core.read_word_32(hw_base + 0x10)?;
            let hw_lr = core.read_word_32(hw_base + 0x14)?;
            let hw_pc = core.read_word_32(hw_base + 0x18)?;
            let hw_xpsr = core.read_word_32(hw_base + 0x1C)?;

            regs.push(SavedRegister { id: RegisterId(0),  value: Some(hw_r0) });
            regs.push(SavedRegister { id: RegisterId(1),  value: Some(hw_r1) });
            regs.push(SavedRegister { id: RegisterId(2),  value: Some(hw_r2) });
            regs.push(SavedRegister { id: RegisterId(3),  value: Some(hw_r3) });
            regs.push(SavedRegister { id: RegisterId(12), value: Some(hw_r12) });
            regs.push(SavedRegister { id: RegisterId(14), value: Some(hw_lr) });
            regs.push(SavedRegister { id: RegisterId(15), value: Some(hw_pc) });
            regs.push(SavedRegister { id: RegisterId(16), value: Some(hw_xpsr) });

            // SP reconstruction: skip SW frame + HW exception frame.
            // Also account for xPSR bit 9 (stack was 8-byte aligned on exception entry).
            let align_adjust = if hw_xpsr & (1 << 9) != 0 { 4u64 } else { 0 };
            let reconstructed_sp = hw_base + hw_frame_size + align_adjust;
            regs.push(SavedRegister {
                id: RegisterId(13),
                value: Some(reconstructed_sp as u32),
            });
        } else {
            // Voluntarily-switched thread: the saved LR is a real code address
            // where the thread will resume. Report it as both LR and PC.
            regs.push(SavedRegister { id: RegisterId(14), value: Some(saved_lr) }); // LR
            regs.push(SavedRegister { id: RegisterId(15), value: Some(saved_lr) }); // PC

            // Caller-saved registers are not part of the voluntary context switch.
            regs.push(SavedRegister { id: RegisterId(0),  value: None }); // r0
            regs.push(SavedRegister { id: RegisterId(1),  value: None }); // r1
            regs.push(SavedRegister { id: RegisterId(2),  value: None }); // r2
            regs.push(SavedRegister { id: RegisterId(3),  value: None }); // r3
            regs.push(SavedRegister { id: RegisterId(12), value: None }); // r12
            regs.push(SavedRegister { id: RegisterId(16), value: None }); // xpsr

            // SP reconstruction: skip only the software frame.
            let reconstructed_sp = sp + self.sw_frame.size;
            regs.push(SavedRegister {
                id: RegisterId(13),
                value: Some(reconstructed_sp as u32),
            });
        }

        Ok(regs)
    }
}
