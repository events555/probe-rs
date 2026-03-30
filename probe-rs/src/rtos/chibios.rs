//! ChibiOS/RT thread awareness.
//!
//! ChibiOS embeds a `chdebug_t` structure in the binary (via `chregistry.h`)
//! that contains all the byte-offsets into the thread struct. This makes
//! ChibiOS uniquely easy to support — no hardcoded offsets, no version-specific
//! tables. We just read the signature and follow where it points.
//!
//! # Detection
//!
//! We look for the `ch_debug` symbol. The first 5 bytes must be `"main\0"`.
//!
//! # Thread enumeration
//!
//! The thread registry is a doubly-linked list rooted at the `rlist` (or `ch`)
//! symbol. We walk `cf_off_newer` pointers until we loop back to the root.
//!
//! # Register recovery
//!
//! Each thread's `port_context.sp` points to a `port_intctx` on its stack.
//! Without FPU: `{r4-r11, lr}` (9 words, 0x24 bytes).
//! With FPU:    `{s16-s31, r4-r11, lr}` (25 words, 0x64 bytes).
//! We detect FPU at runtime by reading the CPACR register.

use super::{read_cstring, ResolvedSymbols, RtosAwareness, RtosThread, SavedRegister, SymbolQuery};
use crate::{Core, Error, RegisterId};
use crate::MemoryInterface;

/// CPACR register address (Coprocessor Access Control Register).
const CPACR_ADDR: u64 = 0xE000_ED88;

/// Thread state names, indexed by the `state` byte in the thread struct.
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
/// This is a packed struct that ChibiOS places in memory to tell debuggers
/// where fields live inside `thread_t`. All values are byte offsets.
#[derive(Debug)]
#[allow(dead_code)] // Fields parsed from the binary signature; not all used yet.
struct ChibiDebugSignature {
    ptr_size: u8,
    thread_size: u8,
    off_prio: u8,
    off_ctx: u8,
    off_newer: u8,
    off_older: u8,
    off_name: u8,
    off_stklimit: u8,
    off_state: u8,
    off_flags: u8,
}

impl ChibiDebugSignature {
    /// Read and validate the signature from target memory.
    fn read_from(core: &mut Core, addr: u64) -> Result<Self, Error> {
        let mut buf = [0u8; 20];
        core.read_8(addr, &mut buf)?;

        // Bytes 0..5: identifier, must be "main\0"
        if &buf[0..5] != b"main\0" {
            return Err(Error::Other(format!(
                "ChibiOS debug signature not found (expected 'main\\0' magic)"
            )));
        }

        // Byte 5: ch_size (size of this struct), must be >= 20
        let struct_size = buf[5];
        if (struct_size as usize) < buf.len() {
            return Err(Error::Other(format!(
                "ChibiOS debug signature too small: {} < {}",
                struct_size,
                buf.len()
            )));
        }

        // Byte 8: pointer size, must be 4 for 32-bit targets
        if buf[8] != 4 {
            return Err(Error::Other(format!(
                "ChibiOS reports pointer size {}, expected 4",
                buf[8]
            )));
        }

        Ok(Self {
            ptr_size: buf[8],
            thread_size: buf[10],
            off_prio: buf[11],
            off_ctx: buf[12],
            off_newer: buf[13],
            off_older: buf[14],
            off_name: buf[15],
            off_stklimit: buf[16],
            off_state: buf[17],
            off_flags: buf[18],
        })
    }
}

/// Whether the Cortex-M FPU is enabled, which changes the context frame size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FpuState {
    Disabled,
    Enabled,
}

impl FpuState {
    /// Check the CPACR register to see if CP10/CP11 have full access.
    fn detect(core: &mut Core) -> Result<Self, Error> {
        let cpacr = core.read_word_32(CPACR_ADDR)?;
        if cpacr & 0x00F0_0000 != 0 {
            Ok(FpuState::Enabled)
        } else {
            Ok(FpuState::Disabled)
        }
    }

    /// Size in bytes of the software-saved context frame (`port_intctx`).
    fn sw_frame_size(self) -> u64 {
        match self {
            // r4-r11, lr = 9 registers * 4 bytes
            FpuState::Disabled => 0x24,
            // s16-s31 (16 regs) + r4-r11, lr (9 regs) = 25 * 4 bytes
            FpuState::Enabled => 0x64,
        }
    }

    /// Byte offset from `sp` to the first general-purpose register (r4).
    fn gpr_offset(self) -> u64 {
        match self {
            FpuState::Disabled => 0x00,
            // Skip the 16 FPU registers (s16-s31)
            FpuState::Enabled => 0x40,
        }
    }
}

/// ChibiOS/RT thread awareness backend.
pub struct ChibiOsAwareness {
    sig: ChibiDebugSignature,
    rlist_addr: u64,
    fpu: FpuState,
}

impl ChibiOsAwareness {
    /// Try to detect ChibiOS on the target.
    pub fn detect(core: &mut Core, symbols: &ResolvedSymbols) -> Option<Box<dyn RtosAwareness>> {
        let ch_debug_addr = *symbols.get("ch_debug")?;

        let sig = ChibiDebugSignature::read_from(core, ch_debug_addr).ok()?;

        // ChibiOS 2.x uses `rlist` directly; ChibiOS 3+ nests it inside `ch`.
        let rlist_addr = symbols
            .get("rlist")
            .or_else(|| symbols.get("ch"))
            .copied()?;

        let fpu = FpuState::detect(core).ok()?;

        Some(Box::new(Self {
            sig,
            rlist_addr,
            fpu,
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
        let rlist = self.rlist_addr;

        // The current thread pointer lives at `rlist + cf_off_name` by ChibiOS
        // convention — the `cf_off_name` offset in the ready list header happens
        // to coincide with the `current` field.
        let current_thread = core.read_word_32(rlist + sig.off_name as u64)? as u64;

        // First pass: validate the doubly-linked list integrity and count threads.
        let mut count = 0usize;
        let mut addr = core.read_word_32(rlist + sig.off_newer as u64)? as u64;
        let mut prev = rlist;

        loop {
            if addr == 0 {
                return Err(Error::Other(format!(
                    "ChibiOS registry: NULL pointer in thread list"
                )));
            }
            if addr == rlist {
                break; // Full loop completed
            }

            // Integrity: check backward pointer matches where we came from
            let older = core.read_word_32(addr + sig.off_older as u64)? as u64;
            if older != prev {
                return Err(Error::Other(format!(
                    "ChibiOS registry: doubly-linked list integrity check failed \
                     (older={:#x}, expected prev={:#x})",
                    older,
                    prev,
                )));
            }

            count += 1;
            if count > MAX_THREADS {
                return Err(Error::Other(format!(
                    "ChibiOS registry: more than {} threads, likely corrupted",
                    MAX_THREADS
                )));
            }

            prev = addr;
            addr = core.read_word_32(addr + sig.off_newer as u64)? as u64;
        }

        // Second pass: collect thread details.
        let mut threads = Vec::with_capacity(count);
        addr = core.read_word_32(rlist + sig.off_newer as u64)? as u64;

        while addr != rlist {
            let name_ptr = core.read_word_32(addr + sig.off_name as u64)? as u64;
            let name = if name_ptr != 0 {
                read_cstring(core, name_ptr, MAX_THREAD_NAME)?
            } else {
                String::new()
            };

            let state_byte = core.read_word_8(addr + sig.off_state as u64)?;
            let state = THREAD_STATES
                .get(state_byte as usize)
                .unwrap_or(&"Unknown")
                .to_string();

            let priority = core.read_word_8(addr + sig.off_prio as u64)?;

            threads.push(RtosThread {
                id: addr,
                name: if name.is_empty() {
                    "unnamed".into()
                } else {
                    name
                },
                state,
                priority,
                is_current: addr == current_thread,
            });

            addr = core.read_word_32(addr + sig.off_newer as u64)? as u64;
        }

        Ok(threads)
    }

    fn thread_registers(
        &self,
        core: &mut Core,
        thread_id: u64,
    ) -> Result<Vec<SavedRegister>, Error> {
        // Read the saved stack pointer from the thread's port_context.
        let sp = core.read_word_32(thread_id + self.sig.off_ctx as u64)? as u64;

        let gpr_base = sp + self.fpu.gpr_offset();

        // Cortex-M register map. ChibiOS only saves callee-saved registers
        // (r4-r11) and LR during context switch. The caller-saved registers
        // (r0-r3, r12) and xpsr are NOT part of the software context frame —
        // they are only in the hardware exception frame for interrupted threads.
        // We explicitly mark them as unavailable.
        let mut regs = vec![
            // Caller-saved: not stacked by ChibiOS context switch.
            SavedRegister { id: RegisterId(0),  value: None }, // r0
            SavedRegister { id: RegisterId(1),  value: None }, // r1
            SavedRegister { id: RegisterId(2),  value: None }, // r2
            SavedRegister { id: RegisterId(3),  value: None }, // r3
            SavedRegister { id: RegisterId(12), value: None }, // r12
            SavedRegister { id: RegisterId(16), value: None }, // xpsr
        ];

        // Callee-saved: read from the software context frame.
        const SW_REGS: &[(u16, u64)] = &[
            (4, 0x00),  // r4
            (5, 0x04),  // r5
            (6, 0x08),  // r6
            (7, 0x0C),  // r7
            (8, 0x10),  // r8
            (9, 0x14),  // r9
            (10, 0x18), // r10
            (11, 0x1C), // r11
            (14, 0x20), // lr
        ];

        for &(reg_num, offset) in SW_REGS {
            let val = core.read_word_32(gpr_base + offset)?;
            regs.push(SavedRegister {
                id: RegisterId(reg_num),
                value: Some(val),
            });
        }

        // The saved LR from the context switch is the return address — where
        // the thread will resume. For voluntarily-switched threads this is a
        // real code address. For threads that were preempted from an ISR, it
        // may be an EXC_RETURN magic value (e.g. 0xFFFFFFFD); proper handling
        // of that case requires parsing the hardware exception frame above.
        //
        // We report the LR as the PC. The DAP stack_trace handler feeds these
        // saved registers into debug_info.unwind() for full DWARF-based call
        // stack unwinding.
        let lr = core.read_word_32(gpr_base + 0x20)?;
        regs.push(SavedRegister {
            id: RegisterId(15), // PC
            value: Some(lr),
        });

        // Reconstruct SP: it pointed to the bottom of the SW frame.
        let reconstructed_sp = sp + self.fpu.sw_frame_size();
        regs.push(SavedRegister {
            id: RegisterId(13), // SP
            value: Some(reconstructed_sp as u32),
        });

        Ok(regs)
    }
}
