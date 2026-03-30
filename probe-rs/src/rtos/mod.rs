//! RTOS thread awareness for embedded debuggers.
//!
//! When debugging firmware that runs an RTOS (ChibiOS, FreeRTOS, Zephyr, etc.),
//! the debugger normally only sees a single execution context per core. This
//! module reads RTOS-internal data structures from target memory to enumerate
//! all threads, their states, and their saved register contexts — so the
//! debugger can present them as separate threads in the DAP protocol.
//!
//! Each RTOS backend implements [`RtosAwareness`] and is auto-detected by
//! looking for known symbols in the loaded ELF.

pub mod chibios;

use crate::{Core, Error, MemoryInterface, RegisterId};
use std::collections::HashMap;

/// A query for a symbol that needs to be resolved from the ELF.
#[derive(Debug, Clone)]
pub struct SymbolQuery {
    /// The symbol name (e.g. `"ch_debug"`, `"pxCurrentTCB"`).
    pub name: &'static str,
    /// If true, the symbol may be absent without failing detection.
    pub optional: bool,
}

/// Resolved symbol addresses, keyed by name.
pub type ResolvedSymbols = HashMap<String, u64>;

/// Information about a single RTOS thread.
#[derive(Debug, Clone)]
pub struct RtosThread {
    /// Unique thread identifier (typically the address of the thread struct).
    pub id: u64,
    /// Human-readable thread name.
    pub name: String,
    /// Human-readable thread state (e.g. "Ready", "WtSem", "Sleeping").
    pub state: String,
    /// Thread priority (RTOS-specific meaning).
    pub priority: u8,
    /// Whether this thread is currently executing on the core.
    pub is_current: bool,
}

/// A saved register value from a non-running thread's context.
#[derive(Debug, Clone, Copy)]
pub struct SavedRegister {
    /// The register identifier (e.g. r4 = RegisterId(4), PC = RegisterId(15)).
    pub id: RegisterId,
    /// The value, or `None` if the register is not saved by the RTOS context
    /// switch (e.g. r0-r3, r12, xpsr on Cortex-M — these are caller-saved and
    /// not part of the software-pushed context frame).
    pub value: Option<u32>,
}

/// Trait that each RTOS backend must implement.
///
/// The lifecycle is:
/// 1. The debugger calls [`RtosAwareness::required_symbols`] to know which ELF
///    symbols to resolve.
/// 2. It resolves them and passes the results to [`RtosAwareness::detect`].
/// 3. If detection succeeds, the returned instance is kept and
///    [`RtosAwareness::threads`] / [`RtosAwareness::thread_registers`] are
///    called on each halt.
pub trait RtosAwareness: Send {
    /// Symbols that must (or may) be present in the ELF for this RTOS.
    fn required_symbols(&self) -> Vec<SymbolQuery>;

    /// Enumerate all threads. Called each time the target halts.
    fn threads(&mut self, core: &mut Core) -> Result<Vec<RtosThread>, Error>;

    /// Read the saved registers for a thread that is *not* currently running.
    /// For the current thread, the debugger should read registers directly from
    /// the core instead.
    fn thread_registers(
        &self,
        core: &mut Core,
        thread_id: u64,
    ) -> Result<Vec<SavedRegister>, Error>;
}

/// Try to detect a known RTOS by resolving symbols and probing target memory.
///
/// Returns the first RTOS backend that successfully detects itself, or `None`.
pub fn detect(core: &mut Core, symbols: &ResolvedSymbols) -> Option<Box<dyn RtosAwareness>> {
    // Try each backend in order. ChibiOS first since it has the strongest
    // detection (magic bytes), reducing false positives.
    if let Some(rtos) = chibios::ChibiOsAwareness::detect(core, symbols) {
        return Some(rtos);
    }

    // Future: FreeRTOS, Zephyr, etc.

    None
}

/// Read a null-terminated C string from target memory.
fn read_cstring(core: &mut Core, addr: u64, max_len: usize) -> Result<String, Error> {
    let mut buf = vec![0u8; max_len];
    core.read_8(addr, &mut buf)?;
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
}
