use super::*;
use std::time::Duration;
use PlaybackMode::*;

const RETURNING: [PlaybackMode; 9] = [
    PingPong,
    Sin,
    EaseIn,
    EaseOut,
    EaseInOut,
    EaseInCubic,
    EaseOutCubic,
    EaseInOutCubic,
    Smootherstep,
];

fn tick(clock: &mut Playback, seconds: f64) -> f32 {
    let mut time = Time::default();
    time.advance_by(Duration::from_secs_f64(seconds));
    clock.step(&time);
    clock.trajectory_progress()
}

fn close(a: f32, b: f32) {
    assert!((a - b).abs() < 2e-6, "{a} != {b}");
}

#[test]
fn returning_curves_retrace_each_leg_and_keep_their_forward_shapes() {
    let quarter_leg = [
        0.25,
        0.14644661,
        0.0625,
        0.4375,
        0.14644661,
        0.015625,
        0.578125,
        0.0625,
        0.103515625,
    ];
    for (mode, quarter) in RETURNING.into_iter().zip(quarter_leg) {
        let mut clock = Playback { mode, ..default() };
        let leg = if mode == Sin { 0.5 } else { 1.0 };
        close(tick(&mut clock, 0.25 * leg), quarter);
        close(tick(&mut clock, 0.75 * leg), 1.);
        assert_eq!(clock.direction, -1., "{mode:?}");
        close(tick(&mut clock, 0.75 * leg), quarter);
        close(tick(&mut clock, 0.25 * leg), 0.);
        close(tick(&mut clock, 13.75 * leg), quarter);
        assert_eq!(clock.direction, -1., "{mode:?}");
        clock.rewind();
        close(tick(&mut clock, 2.0 * leg), 0.);
        assert_eq!(clock.direction, 1., "{mode:?}");
    }
}

#[test]
fn elapsed_time_survives_irregular_frames_and_reversed_speed() {
    for mode in RETURNING {
        let start = Playback {
            mode,
            progress: 0.35,
            direction: -1.,
            speed: 0.7,
        };
        let mut fine = start;
        let mut coarse = start;
        let frames = [17, 33, 9, 21, 8, 12];
        for _ in 0..71 {
            for millis in frames {
                tick(&mut fine, f64::from(millis) / 1000.);
            }
        }
        tick(&mut coarse, 7.1);
        assert_eq!(fine.direction, coarse.direction, "{mode:?}");
        assert!(
            (fine.trajectory_progress() - coarse.trajectory_progress()).abs() < 1e-4,
            "{mode:?} {fine:?} {coarse:?}"
        );
        coarse.speed = -coarse.speed;
        tick(&mut coarse, 7.1);
        close(coarse.trajectory_progress(), start.trajectory_progress());
        assert_eq!(coarse.direction, start.direction);
    }
}

#[test]
fn smooth_turnarounds_and_stops_stay_inside_the_authored_path() {
    for mode in [EaseInOut, EaseInOutCubic, Smootherstep] {
        let mut last = 0.;
        for i in 0..=4096 {
            let pos = mode.map_progress(i as f32 / 4096.);
            assert!((last..=1.).contains(&pos), "{mode:?} {i} {pos}");
            last = pos;
        }
        let mut clock = Playback { mode, ..default() };
        let before = tick(&mut clock, 0.99);
        let after = tick(&mut clock, 0.02);
        close(before, after);
        assert!(after > 0.9997);
        let before = tick(&mut clock, 0.98);
        close(before, tick(&mut clock, 0.02));
        assert!(before < 0.0003);
    }
    assert!(Smootherstep.map_progress(0.01) < 0.00001);
}

#[test]
fn switching_curves_or_pausing_preserves_pose_and_return_leg() {
    for mode in RETURNING {
        for position in [0., 0.001, 0.2, 0.5, 0.85, 0.999, 1.] {
            let mut clock = Playback {
                progress: position,
                direction: -1.,
                ..default()
            };
            for next in [mode, Still, Smootherstep, EaseOut, mode] {
                clock.set_mode(next);
                close(clock.trajectory_progress(), position);
                assert_eq!(clock.direction, -1.);
            }
            clock.speed = 0.;
            close(tick(&mut clock, 100.), position);
        }
    }
}

#[test]
fn once_holds_loop_restarts_and_still_holds_the_requested_position() {
    let mut clock = Playback {
        mode: Once,
        progress: 0.75,
        ..default()
    };
    close(tick(&mut clock, 1.5), 1.);
    close(tick(&mut clock, 0.125), 1.);
    clock.set_mode(Loop);
    close(tick(&mut clock, 3.125), 0.125);
    close(tick(&mut clock, 0.875), 0.);
    clock.set_mode(Still);
    clock.progress = 0.75;
    close(tick(&mut clock, 10.), 0.75);
}
