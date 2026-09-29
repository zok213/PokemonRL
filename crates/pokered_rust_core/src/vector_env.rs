//! vector_env.rs — Rayon-Powered Multithreaded Vectorized Environment
//! Executes batched simulation steps across CPU threads with zero GIL and zero serialization.

use crate::lr35902::{GameBoyInstance, wram_addr};
use rayon::prelude::*;

pub struct VectorEnvironment {
    pub instances: Vec<GameBoyInstance>,
    pub num_envs: usize,
}

impl VectorEnvironment {
    pub fn new(num_envs: usize) -> Self {
        let instances = (0..num_envs).map(|_| GameBoyInstance::new()).collect();
        Self { instances, num_envs }
    }

    /// Parallel step across all environments using Rayon work-stealing threadpool
    pub fn step_batch(
        &mut self,
        actions: &[u8],
        screens_out: &mut [u8],     // [B, 3, 72, 80]
        wrams_out: &mut [f32],       // [B, 64]
        rewards_out: &mut [f32],     // [B]
        dones_out: &mut [u8],        // [B]
        masks_out: &mut [u8],        // [B, 8]
    ) {
        assert_eq!(actions.len(), self.num_envs);

        // Parallel update via Rayon
        let num_envs = self.num_envs;
        let instances = &mut self.instances;

        // Step each emulator in parallel
        let results: Vec<(f32, bool, u8, u8, u8, u8, u8)> = instances
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(inst, &act)| {
                let (reward, done, mask) = inst.step(act);
                (reward, done, mask, inst.cur_map, inst.x_pos, inst.y_pos, inst.badges)
            })
            .collect();

        // Write contiguous output buffers
        for (i, &(reward, done, mask_byte, map_n, x, y, badges)) in results.iter().enumerate() {
            rewards_out[i] = reward;
            dones_out[i] = if done { 1 } else { 0 };

            // Unpack 8 action bits
            for a in 0..8 {
                masks_out[i * 8 + a] = (mask_byte >> a) & 1;
            }

            // Fill WRAM feature slice
            let w_base = i * 64;
            wrams_out[w_base + 0] = (map_n as f32) / 255.0;
            wrams_out[w_base + 1] = (x as f32) / 255.0;
            wrams_out[w_base + 2] = (y as f32) / 255.0;
            wrams_out[w_base + 3] = (badges as f32) / 8.0;
        }
    }
}
