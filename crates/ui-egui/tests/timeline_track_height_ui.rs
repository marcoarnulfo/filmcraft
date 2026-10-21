//! Headless UI tests of changing track heights by dragging the lines between track headers,
//! including the outer edges: the top of the highest video track and the bottom of the last audio
//! track.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_project::TrackKind;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
}

impl Driver {
    fn demo() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let harness = Harness::builder().with_size(egui::vec2(1600.0, 1400.0)).with_max_steps(10_000).build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx };
        d.frames(4);
        // short enough that every track is in view, with empty space above and below them
        d.ok("ui.set", json!({"timeline": {"videoTrackHeight": 24.0, "audioTrackHeight": 24.0}}));
        d.frames(3);
        d
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            eframe::App::raw_input_hook(self.harness.state_mut(), &ctx, &mut raw);
            *self.harness.input_mut() = raw;
            self.harness.step();
        }
    }

    fn ok(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(req).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(v) = reply.try_recv() {
                assert_eq!(v["ok"], json!(true), "{method} {params} failed: {v}");
                return v["result"].clone();
            }
        }
        panic!("no reply to {method} {params}");
    }

    fn app(&self) -> &FilmcraftApp {
        self.harness.state()
    }

    /// Drag from the middle of the element `id` by `dy` points.
    fn drag(&mut self, id: &str, fy: f32, dy: f32) {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        let e = v.as_array().unwrap().iter().find(|e| e["id"] == json!(id)).unwrap_or_else(|| panic!("no element {id}")).clone();
        let r: Vec<f32> = e["rect"].as_array().unwrap().iter().map(|n| n.as_f64().unwrap() as f32).collect();
        let (x, y) = (r[0] + r[2] / 2.0, r[1] + r[3] * fy);
        self.ok("ui.drag", json!({"from": {"x": x, "y": y}, "to": {"x": x, "y": y + dy}}));
        self.frames(3);
    }

    /// The height of each row of `kind`, in track order (V1 or A1 first).
    fn row_heights(&self, kind: TrackKind) -> Vec<f32> {
        let layout = self.app().tl.layout.as_ref().expect("timeline layout");
        let mut rows: Vec<_> = layout.rows.iter().filter(|r| r.kind == kind).collect();
        rows.sort_by_key(|r| r.index);
        rows.iter().map(|r| r.rect.height()).collect()
    }

    fn labels(&self) -> (String, String) {
        let seq = self.app().session.active_sequence().unwrap();
        (format!("V{}", seq.video_tracks.len()), format!("A{}", seq.audio_tracks.len()))
    }
}

#[test]
fn the_top_edge_of_the_highest_video_track_resizes_that_track() {
    let mut d = Driver::demo();
    let (top, _) = d.labels();
    let (v0, a0) = (d.row_heights(TrackKind::Video), d.row_heights(TrackKind::Audio));
    let n = v0.len() - 1;
    // dragging the outer edge up makes the track taller, as the line below it does
    d.drag(&format!("timeline.track.{top}.resizeTop"), 0.5, -15.0);
    let v1 = d.row_heights(TrackKind::Video);
    assert!(v1[n] > v0[n] + 5.0, "{top} grows: {v0:?} -> {v1:?}");
    assert_eq!(v1[..n], v0[..n], "the video tracks below it are left alone");
    assert_eq!(d.row_heights(TrackKind::Audio), a0, "audio tracks are left alone");
    // and down makes it shorter again
    d.drag(&format!("timeline.track.{top}.resizeTop"), 0.5, 15.0);
    assert!(d.row_heights(TrackKind::Video)[n] < v1[n] - 5.0);
    // the line below it still works the same way
    let v2 = d.row_heights(TrackKind::Video)[n];
    d.drag(&format!("timeline.track.{top}.resize"), 0.5, -15.0);
    assert!(d.row_heights(TrackKind::Video)[n] > v2 + 5.0);
}

#[test]
fn the_bottom_edge_of_the_last_audio_track_resizes_that_track_from_either_side_of_the_line() {
    let mut d = Driver::demo();
    let (_, last) = d.labels();
    let id = format!("timeline.track.{last}.resize");
    let (v0, a0) = (d.row_heights(TrackKind::Video), d.row_heights(TrackKind::Audio));
    let n = a0.len() - 1;
    // from just past the line, in the empty space below the last track
    d.drag(&id, 0.9, 15.0);
    let a1 = d.row_heights(TrackKind::Audio);
    assert!(a1[n] > a0[n] + 5.0, "{last} grows: {a0:?} -> {a1:?}");
    assert_eq!(a1[..n], a0[..n], "the audio tracks above it are left alone");
    assert_eq!(d.row_heights(TrackKind::Video), v0, "video tracks are left alone");
    // and from just inside it
    d.drag(&id, 0.1, -15.0);
    assert!(d.row_heights(TrackKind::Audio)[n] < a1[n] - 5.0);
}

#[test]
fn dragging_a_line_resizes_only_the_track_it_belongs_to() {
    let mut d = Driver::demo();
    let (v0, a0) = (d.row_heights(TrackKind::Video), d.row_heights(TrackKind::Audio));
    assert!(v0.len() >= 2 && a0.len() >= 2, "the demo has several tracks of each kind: {v0:?} {a0:?}");
    // V1's line is below it: up makes V1 taller
    d.drag("timeline.track.V1.resize", 0.5, -15.0);
    let v1 = d.row_heights(TrackKind::Video);
    assert!(v1[0] > v0[0] + 5.0, "V1 grows: {v0:?} -> {v1:?}");
    assert_eq!(v1[1..], v0[1..], "the other video tracks keep their height");
    assert_eq!(d.row_heights(TrackKind::Audio), a0, "audio tracks are left alone");
    // A1's line is below it: down makes A1 taller
    d.drag("timeline.track.A1.resize", 0.5, 15.0);
    let a1 = d.row_heights(TrackKind::Audio);
    assert!(a1[0] > a0[0] + 5.0, "A1 grows: {a0:?} -> {a1:?}");
    assert_eq!(a1[1..], a0[1..], "the other audio tracks keep their height");
    assert_eq!(d.row_heights(TrackKind::Video), v1, "video tracks are left alone");
    // setting the height of all the video tracks includes the one resized on its own
    d.ok("ui.set", json!({"timeline": {"videoTrackHeight": 30.0}}));
    d.frames(2);
    assert!(d.row_heights(TrackKind::Video).iter().all(|h| *h == 30.0), "{:?}", d.row_heights(TrackKind::Video));
    assert_eq!(d.row_heights(TrackKind::Audio), a1, "audio tracks are left alone");
}

#[test]
fn expand_all_tracks_pressed_again_gives_back_a_track_resized_on_its_own() {
    let mut d = Driver::demo();
    // row heights come from rects, so they are compared to a tenth of a point
    let rows = |d: &Driver| [TrackKind::Video, TrackKind::Audio].map(|k| d.row_heights(k).iter().map(|h| (h * 10.0).round() / 10.0).collect::<Vec<_>>());
    d.drag("timeline.track.V1.resize", 0.5, -15.0);
    d.drag("timeline.track.A1.resize", 0.5, 15.0);
    let before = rows(&d);
    let [v0, a0] = &before;
    assert!(v0[0] > v0[1] && a0[0] > a0[1], "V1 and A1 resized on their own: {v0:?} {a0:?}");
    for key in ["Shift+=", "Shift+-"] {
        d.ok("ui.key", json!({"key": key}));
        d.frames(3);
        let [v1, a1] = rows(&d);
        assert!(v1.iter().chain(&a1).all(|h| *h == v1[0]) && v1[0] != v0[0], "{key}: every track at one height: {v1:?} {a1:?}");
        // pressing again gives every track the height it had, the ones resized on their own too
        d.ok("ui.key", json!({"key": key}));
        d.frames(3);
        assert_eq!(rows(&d), before, "{key} pressed again");
    }
}

#[test]
fn a_track_resized_on_its_own_keeps_its_height_in_the_saved_project() {
    let dir = std::env::temp_dir().join(format!("filmcraft-track-heights-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("heights.fcproj").to_string_lossy().to_string();
    let mut d = Driver::demo();
    d.drag("timeline.track.V1.resize", 0.5, -15.0);
    d.drag("timeline.track.A1.resize", 0.5, 15.0);
    let rows = |d: &Driver| [TrackKind::Video, TrackKind::Audio].map(|k| d.row_heights(k).iter().map(|h| (h * 10.0).round() / 10.0).collect::<Vec<_>>());
    let before = rows(&d);
    assert!(before[0][0] > before[0][1] && before[1][0] > before[1][1], "V1 and A1 resized on their own: {before:?}");
    d.ok("engine.execute", json!({"command": "file.saveAs", "params": {"path": path}}));

    // another window of the app opens the file
    let mut e = Driver::demo();
    e.ok("engine.execute", json!({"command": "file.open", "params": {"path": path}}));
    e.frames(3);
    assert_eq!(rows(&e), before);
    let _ = std::fs::remove_dir_all(&dir);
}
