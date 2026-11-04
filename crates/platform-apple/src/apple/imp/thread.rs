//! The calling thread's quality-of-service class.

use std::ffi::c_int;

/// `QOS_CLASS_USER_INTERACTIVE` from `<sys/qos.h>`.
const USER_INTERACTIVE: u32 = 0x21;

unsafe extern "C" {
    fn pthread_set_qos_class_self_np(class: u32, relative_priority: c_int) -> c_int;
}

/// Run the calling thread in the user-interactive class, the class of the
/// main thread. A thread the interface waits on needs it, or the system
/// reports a priority inversion and may run it late. Returns false when the
/// system refused.
pub fn run_at_interface_priority() -> bool {
    // SAFETY: the call takes two integers and changes only the calling
    // thread's class.
    unsafe { pthread_set_qos_class_self_np(USER_INTERACTIVE, 0) == 0 }
}

#[cfg(test)]
mod tests {
    use super::run_at_interface_priority;

    #[test]
    fn a_new_thread_takes_the_interface_class() {
        assert!(
            std::thread::spawn(run_at_interface_priority)
                .join()
                .unwrap()
        );
    }
}
