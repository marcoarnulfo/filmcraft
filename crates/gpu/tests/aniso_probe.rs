//! INVESTIGATION ONLY (not for a PR): quality and cost of the minification footprint.
//!
//! quality: `cargo test -p filmcraft-gpu --test aniso_probe quality -- --nocapture --test-threads=1`
//! (the dev profile builds gpu, render and frame at opt-level 3 on this branch; `--release` works too)
//! timing:  `... --test aniso_probe timing -- --nocapture --test-threads=1`
//! Env: FC_FOOTPRINT=axis (prototype) or unset (main), FC_TAPCAP=4|8.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Instant;

use filmcraft_frame::{Chroma, PixelData, VideoFrame};
use filmcraft_geom::Affine;
use filmcraft_gpu::GpuCompositor;
use filmcraft_render::Blend;
use filmcraft_render::plan::{FramePlan, PlanLayer, execute_cpu};
use rayon::prelude::*;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    eprintln!("adapter: {:?}", adapter.get_info().name);
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

fn srgb_to_lin(c: f64) -> f64 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}
fn lin_to_srgb(l: f64) -> f64 {
    let l = l.clamp(0.0, 1.0);
    if l <= 0.0031308 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 }
}

/// Procedural gray patterns (8-bit sRGB codes), so anyone can regenerate them.
fn pattern(kind: &str, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n * n];
    for y in 0..n {
        for x in 0..n {
            let g = match kind {
                // zone plate: local frequency r / n cycles per pixel (Nyquist at r = n / 2)
                "zone" => {
                    let (dx, dy) = (x as f64 + 0.5 - n as f64 / 2.0, y as f64 + 0.5 - n as f64 / 2.0);
                    0.5 + 0.5 * (std::f64::consts::PI * (dx * dx + dy * dy) / n as f64).cos()
                }
                // hashed value noise, one value per pixel (fine texture: foliage, gravel, fabric)
                _ => {
                    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
                    h ^= h >> 31;
                    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    h ^= h >> 29;
                    (h >> 40) as f64 / (1u64 << 24) as f64
                }
            };
            v[y * n + x] = (g * 255.0).round() as u8;
        }
    }
    v
}

/// Reference: box average of the bilinearly reconstructed source over each output pixel's exact
/// footprint (the parallelogram the output pixel maps to), K x K samples, in linear light.
fn reference(lin: &[f64], n: usize, m: &Affine, ow: usize, oh: usize) -> Vec<Option<u8>> {
    let inv = m.inverse().unwrap();
    let (jx, jy) = ((inv.a, inv.b), (inv.c, inv.d));
    let lj = (jx.0.hypot(jx.1)).max(jy.0.hypot(jy.1));
    let k = ((lj * 6.0).ceil() as usize).clamp(8, 64);
    let bil = |u: f64, v: f64| {
        let (q, r) = (u - 0.5, v - 0.5);
        let (i, j) = (q.floor(), r.floor());
        let (fx, fy) = (q - i, r - j);
        let at = |a: f64, b: f64| lin[(b.clamp(0.0, n as f64 - 1.0) as usize) * n + a.clamp(0.0, n as f64 - 1.0) as usize];
        let top = at(i, j) * (1.0 - fx) + at(i + 1.0, j) * fx;
        let bot = at(i, j + 1.0) * (1.0 - fx) + at(i + 1.0, j + 1.0) * fx;
        top * (1.0 - fy) + bot * fy
    };
    (0..ow * oh)
        .into_par_iter()
        .map(|p| {
            let (x, y) = ((p % ow) as f64 + 0.5, (p / ow) as f64 + 0.5);
            let (u, v) = (inv.a * x + inv.c * y + inv.e, inv.b * x + inv.d * y + inv.f);
            // only pixels whose whole footprint lies inside the source (no edge effects)
            let rx = (jx.0.abs() + jy.0.abs()) / 2.0 + 1.0;
            let ry = (jx.1.abs() + jy.1.abs()) / 2.0 + 1.0;
            if u - rx < 0.0 || v - ry < 0.0 || u + rx > n as f64 || v + ry > n as f64 {
                return None;
            }
            let mut acc = 0.0;
            for j in 0..k {
                for i in 0..k {
                    let (a, b) = ((i as f64 + 0.5) / k as f64 - 0.5, (j as f64 + 0.5) / k as f64 - 0.5);
                    acc += bil(u + a * jx.0 + b * jy.0, v + a * jx.1 + b * jy.1);
                }
            }
            Some((lin_to_srgb(acc / (k * k) as f64) * 255.0).round() as u8)
        })
        .collect()
}

fn stats(img: &[u8], refr: &[Option<u8>]) -> (f64, u32, u32, f64) {
    let mut d: Vec<u32> = img.chunks(4).zip(refr).filter_map(|(p, r)| r.map(|r| (p[0] as i32 - r as i32).unsigned_abs())).collect();
    d.sort_unstable();
    let mean = d.iter().sum::<u32>() as f64 / d.len() as f64;
    let mse = d.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / d.len() as f64;
    let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
    (mean, d[d.len() * 99 / 100], d[d.len() - 1], psnr)
}

#[test]
fn quality() {
    let Some((dev, q)) = device() else { return };
    let mode = std::env::var("FC_FOOTPRINT").unwrap_or_else(|_| "iso(main)".into());
    let cap = std::env::var("FC_TAPCAP").unwrap_or_else(|_| "4".into());
    let mut c = GpuCompositor::new(&dev, &q);
    let n = 512usize;
    // (label, sx, sy, rotation degrees)
    let cases: &[(&str, f64, f64, f64)] = &[
        ("2 x 0.5", 2.0, 0.5, 0.0),
        ("0.5 x 2", 0.5, 2.0, 0.0),
        ("1 x 0.25", 1.0, 0.25, 0.0),
        ("0.5 x 0.25", 0.5, 0.25, 0.0),
        ("0.5 x 0.5", 0.5, 0.5, 0.0),
        ("0.5 x 0.5 rot 30", 0.5, 0.5, 30.0),
        ("0.25 x 0.25", 0.25, 0.25, 0.0),
        ("0.125 x 0.125", 0.125, 0.125, 0.0),
        ("0.1 x 0.1", 0.1, 0.1, 0.0),
    ];
    for kind in ["zone", "noise"] {
        let codes = pattern(kind, n);
        let lin: Vec<f64> = codes.iter().map(|c| srgb_to_lin(*c as f64 / 255.0)).collect();
        let rgba: Vec<u8> = codes.iter().flat_map(|c| [*c, *c, *c, 255]).collect();
        let frame = Arc::new(VideoFrame::rgba8(n as u32, n as u32, rgba));
        for (label, sx, sy, rot) in cases {
            let core = Affine::rotate_deg(*rot).then_apply(&Affine::scale(*sx, *sy)).then_apply(&Affine::translate(-(n as f64) / 2.0, -(n as f64) / 2.0));
            let b = core.bounds(&filmcraft_geom::Rect::new(0.0, 0.0, n as f64, n as f64));
            let (ow, oh) = ((b.w.ceil() as usize + 8).max(16), (b.h.ceil() as usize + 8).max(16));
            let m = Affine::translate(ow as f64 / 2.0 + 0.37, oh as f64 / 2.0 + 0.21).then_apply(&core);
            let plan = FramePlan::Layers { width: ow, height: oh, layers: vec![PlanLayer::new(frame.clone(), m, 1.0, Blend::Normal)] };
            let refr = reference(&lin, n, &m, ow, oh);
            c.composite(&plan);
            let (_, _, gpu) = c.read_output().unwrap();
            let cpu = execute_cpu(&plan).over_black_rgba8();
            let (gm, gp, gx, gs) = stats(&gpu, &refr);
            let (cm, cp, cx, cs) = stats(&cpu, &refr);
            println!(
                "Q|{mode}|cap{cap}|{kind}|{label}|GPU mean {gm:.2} p99 {gp} max {gx} psnr {gs:.1}|CPU mean {cm:.2} p99 {cp} max {cx} psnr {cs:.1}"
            );
        }
    }
}

fn camera_4k() -> Arc<VideoFrame> {
    let (w, h) = (3840usize, 2160usize);
    let noise = pattern("noise", 2160);
    let y: Vec<u8> = (0..w * h).map(|i| (16 + (noise[(i / w) * 2160 + (i % w) % 2160] as usize * 219 / 255)) as u8).collect();
    let (cw, ch) = (w / 2, h / 2);
    let u: Vec<u8> = (0..cw * ch).map(|i| (100 + (i % cw) * 40 / cw) as u8).collect();
    let v: Vec<u8> = (0..cw * ch).map(|i| (150 - (i / cw) * 40 / ch) as u8).collect();
    Arc::new(VideoFrame {
        width: w as u32,
        height: h as u32,
        data: PixelData::Yuv8 { planes: [Arc::new(y), Arc::new(u), Arc::new(v)], chroma: Chroma::C420, alpha: None },
        color: filmcraft_color::ColorInfo::REC709,
        par: (1, 1),
        pts: Default::default(),
    })
}

#[test]
fn timing() {
    let Some((dev, q)) = device() else { return };
    let mode = std::env::var("FC_FOOTPRINT").unwrap_or_else(|_| "iso(main)".into());
    let cap = std::env::var("FC_TAPCAP").unwrap_or_else(|_| "4".into());
    let mut c = GpuCompositor::new(&dev, &q);
    let frame = camera_4k();
    let (ow, oh) = (1920usize, 1080usize);
    let centre = |sx: f64, sy: f64| {
        Affine::translate(960.0, 540.0).then_apply(&Affine::scale(sx, sy)).then_apply(&Affine::translate(-1920.0, -1080.0))
    };
    let cases: Vec<(&str, Affine)> = vec![
        ("4K fit 1080p (0.5 x 0.5)", centre(0.5, 0.5)),
        ("4K squash (0.5 x 0.25)", centre(0.5, 0.25)),
        ("4K stretch (1 x 0.25)", centre(1.0, 0.25)),
        ("4K PIP 1/8 (0.125 x 0.125)", centre(0.125, 0.125)),
        ("4K x 3 layers at 0.5", centre(0.5, 0.5)),
    ];
    for (label, m) in cases {
        let layers = if label.contains("3 layers") { 3 } else { 1 };
        let plan = FramePlan::Layers { width: ow, height: oh, layers: (0..layers).map(|_| PlanLayer::new(frame.clone(), m, 0.8, Blend::Normal)).collect() };
        for _ in 0..5 {
            c.composite(&plan);
            dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        }
        let mut t: Vec<f64> = (0..60)
            .map(|_| {
                let t0 = Instant::now();
                c.composite(&plan);
                dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
                t0.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("T|{mode}|cap{cap}|{label}|median {:.3} ms|p10 {:.3}|p90 {:.3}", t[30], t[6], t[54]);
    }
}
