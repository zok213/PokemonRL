//! lib.rs — C-ABI & PyO3 Exports for pokered_rust_core
//! Provides `step_vectorized` C-ABI function callable from Python ctypes or C++.

pub mod lr35902;
pub mod vector_env;

use vector_env::VectorEnvironment;
use std::sync::Mutex;
use std::sync::OnceLock;

static GLOBAL_ENV: OnceLock<Mutex<VectorEnvironment>> = OnceLock::new();

fn get_or_init_env(num_envs: usize) -> &'static Mutex<VectorEnvironment> {
    GLOBAL_ENV.get_or_init(|| Mutex::new(VectorEnvironment::new(num_envs)))
}

/// C-ABI exported entrypoint for Python ctypes / C++ zero-copy vectorization
#[no_mangle]
pub unsafe extern "C" fn step_vectorized(
    actions_ptr: *const u8,
    screens_ptr: *mut u8,
    wrams_ptr: *mut f32,
    rewards_ptr: *mut f32,
    dones_ptr: *mut u8,
    masks_ptr: *mut u8,
    batch_size: i32,
) {
    if actions_ptr.is_null() || batch_size <= 0 {
        return;
    }

    let b = batch_size as usize;
    let env_mutex = get_or_init_env(b);
    let mut env = env_mutex.lock().unwrap();

    let actions = std::slice::from_raw_parts(actions_ptr, b);
    let screens = std::slice::from_raw_parts_mut(screens_ptr, b * 3 * 72 * 80);
    let wrams = std::slice::from_raw_parts_mut(wrams_ptr, b * 64);
    let rewards = std::slice::from_raw_parts_mut(rewards_ptr, b);
    let dones = std::slice::from_raw_parts_mut(dones_ptr, b);
    let masks = std::slice::from_raw_parts_mut(masks_ptr, b * 8);

    env.step_batch(actions, screens, wrams, rewards, dones, masks);
}
