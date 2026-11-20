"""Record Pong Actor-Critic using the project's reconstructed compatible setup.

Run from any working directory with the existing mlds2018-hw4 interpreter.
The historical modern/ imports are supplied by the verified modern_pg modules.
Actor-Critic make_action() and init_game_setting() are used directly.
"""

import argparse
import gc
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modern_pg"))
sys.path.insert(0, str(ROOT / "modern_pg_improved"))
from environment import Environment, make_env
from pg_agent import Agent_PG
from pg_policy import PongActor

CHECKPOINT = ROOT / "modern_pg_improved/checkpoints/pong_pg_latest.pt"
SNAPSHOT = ROOT / "modern_pg_improved/checkpoints/pong_pg_10000.pt"
OUTPUT = ROOT / "videos/pong_actor_critic_latest.mp4"
BREAKOUT = ROOT / "videos/breakout_double_dqn_latest.mp4"
EXPECTED_HASH = "a2617a925f676caaf7f6ca0962d925b73eb2e6e26deeb8de2dd34b4be0152786"
FPS = 60
SIZE = (1280, 720)


def file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def weight_digest(actor):
    digest = hashlib.sha256()
    for value in actor.state_dict().values():
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_compatible_environment(render_mode):
    # Supply the missing historical import using the current PG environment
    # factory. Its documented settings remain explicit in videos/README.md.
    env = Environment.__new__(Environment)
    env.env = make_env(
        "Pong", wrapped=False, test=True, render_mode=render_mode,
        max_episode_steps=27000,
    )
    env.action_space = env.env.action_space
    env.observation_space = env.env.observation_space
    env._seed = None
    env.last_reset_info = {}
    meanings = tuple(env.env.unwrapped.get_action_meanings())
    if meanings != ("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"):
        env.close()
        raise RuntimeError(f"Unexpected Pong action mapping: {meanings}")
    return env


def make_inference_agent(env, actor, seed):
    # Restore only the fields used by the original inference methods.
    agent = Agent_PG.__new__(Agent_PG)
    agent.env = env
    agent.args = None
    agent.actor = actor
    agent.previous_frame = None
    agent.action_map = [2, 3]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    return agent


class CaptureALEFrames:
    """Delegate ALE calls and capture after each existing one-frame act().

    Gymnasium itself still samples and executes its original random frame skip.
    This proxy observes those calls and forwards every original argument/result.
    """

    def __init__(self, base_env, consume):
        self.original = base_env.ale
        self.base_env = base_env
        self.consume = consume
        self.frames = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def act(self, *args, **kwargs):
        before = self.original.getFrameNumber()
        reward = self.original.act(*args, **kwargs)
        elapsed = self.original.getFrameNumber() - before
        if elapsed != 1:
            raise RuntimeError(f"Expected one frame per ALE act(), found {elapsed}")
        rgb = self.base_env.render()
        if rgb.shape != (210, 160, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f"Unexpected raw RGB: {rgb.shape}, {rgb.dtype}")
        self.frames += 1
        self.consume(rgb)
        return reward


def validate_actor_input(module, args):
    state = args[0]
    if state.shape != (80, 80) or state.dtype != torch.float32:
        raise RuntimeError(f"Unexpected Actor input: {state.shape}, {state.dtype}")
    if not torch.isfinite(state).all() or state.min() < -1 or state.max() > 1:
        raise RuntimeError("Actor frame difference is outside [-1, 1]")


def verify_capture_equivalence(actor, seed):
    traces = []
    for captured in (False, True):
        env = make_compatible_environment("rgb_array" if captured else None)
        proxy = None
        try:
            agent = make_inference_agent(env, actor, seed)
            observation = env.reset()
            agent.init_game_setting()
            if captured:
                proxy = CaptureALEFrames(env.env.unwrapped, lambda rgb: None)
                env.env.unwrapped.ale = proxy
            trace = []
            for _ in range(64):
                action = agent.make_action(observation, test=True)
                if action not in (2, 3) or not env.action_space.contains(action):
                    raise RuntimeError(f"Illegal policy action: {action}")
                observation, reward, done, info = env.step(action)
                trace.append((action, float(reward), done,
                              hashlib.sha256(observation.tobytes()).hexdigest(),
                              hashlib.sha256(agent.previous_frame.tobytes()).hexdigest()))
                if done:
                    break
            traces.append(trace)
        finally:
            if proxy is not None:
                env.env.unwrapped.ale = proxy.original
            env.close()
    if traces[0] != traces[1]:
        raise RuntimeError("RGB capture changed the compatible inference trajectory")


def validate_video(path, expected_frames):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Cannot open encoded MP4")
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    count, changed = 0, 0
    previous = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            game = frame[45:675, 400:880]
            if float(game.std()) < 1:
                raise RuntimeError(f"Black/corrupt game image at frame {count}")
            if previous is not None and np.any(game != previous):
                changed += 1
            previous = game
            count += 1
    finally:
        capture.release()
    if size != SIZE or abs(fps - FPS) > 0.01 or count != expected_frames:
        raise RuntimeError(f"Incorrect video metadata: {size=}, {fps=}, {count=}")
    if changed < count // 4:
        raise RuntimeError("Too few changing game frames")
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path),
         "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    if "Video: h264" not in result.stderr or "yuv420p" not in result.stderr:
        raise RuntimeError("Expected H.264 / yuv420p encoding")
    return dict(decoded_frames=count, fps=fps, size=list(size), changed_frames=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seconds", type=int, default=60)
    options = parser.parse_args()
    if not 30 <= options.seconds <= 60:
        parser.error("Choose a duration between 30 and 60 seconds")
    temporary = OUTPUT.with_name(OUTPUT.stem + ".partial.mp4")
    if OUTPUT.exists() or temporary.exists():
        raise FileExistsError("A Pong output already exists")
    before_hashes = {path: file_digest(path) for path in (CHECKPOINT, SNAPSHOT, BREAKOUT)}
    if before_hashes[CHECKPOINT] != EXPECTED_HASH or before_hashes[SNAPSHOT] != EXPECTED_HASH:
        raise RuntimeError("Actor-Critic checkpoint differs from the verified 10000-episode snapshot")
    torch.set_num_threads(1)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if int(checkpoint["episodes_completed"]) != 10000 or int(checkpoint["total_steps"]) != 45211537:
        raise RuntimeError("Incorrect Actor-Critic training node")
    if "critic_model" not in checkpoint or checkpoint["critic_model"]["fc2.weight"].shape != (1, 256):
        raise RuntimeError("Expected the Actor-Critic checkpoint with its trained Critic")
    actor = PongActor()
    actor.load_state_dict(checkpoint["model"], strict=True)
    actor.eval().requires_grad_(False)
    del checkpoint
    gc.collect()
    initial_weights = weight_digest(actor)
    input_hook = actor.register_forward_pre_hook(validate_actor_input)

    OUTPUT.parent.mkdir(exist_ok=True)
    canvas = Image.new("RGB", SIZE, (14, 27, 45))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 25)
    title = f"Pong Actor-Critic   /   10,000 episodes   /   Eval seed {options.seed}"
    bounds = draw.textbbox((0, 0), title, font=font)
    draw.text(((SIZE[0] - bounds[2]) // 2, 7), title, fill=(235, 242, 248), font=font)
    small_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
    note = "Reconstructed compatible environment"
    bounds = draw.textbbox((0, 0), note, font=small_font)
    draw.text(((SIZE[0] - bounds[2]) // 2, 691), note, fill=(200, 214, 229), font=small_font)
    frame_limit = options.seconds * FPS
    recorded = 0
    rgb_hash = hashlib.sha256()
    writer = imageio_ffmpeg.write_frames(
        str(temporary), SIZE, fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
        output_params=["-preset", "fast", "-crf", "18", "-movflags", "+faststart"],
    )

    def consume(rgb):
        nonlocal recorded
        if recorded >= frame_limit:
            return
        rgb_hash.update(rgb.tobytes())
        canvas.paste(Image.fromarray(rgb).resize((480, 630), Image.Resampling.NEAREST), (400, 45))
        writer.send(np.asarray(canvas))
        recorded += 1

    env, proxy = None, None
    try:
        with torch.inference_mode():
            verify_capture_equivalence(actor, options.seed)
            print("64-decision capture equivalence check PASS", flush=True)
            env = make_compatible_environment("rgb_array")
            agent = make_inference_agent(env, actor, options.seed)
            observation = env.reset()
            agent.init_game_setting()
            writer.send(None)
            proxy = CaptureALEFrames(env.env.unwrapped, consume)
            env.env.unwrapped.ale = proxy
            actions = Counter()
            rewards = []
            skip_counts = Counter()
            done = False
            while recorded < frame_limit and not done:
                action = agent.make_action(observation, test=True)
                if action not in (2, 3) or not env.action_space.contains(action):
                    raise RuntimeError(f"Illegal policy action: {action}")
                actions[action] += 1
                before = proxy.frames
                observation, reward, done, info = env.step(action)
                skip_counts[proxy.frames - before] += 1
                if reward:
                    rewards.append(float(reward))
    finally:
        if proxy is not None:
            env.env.unwrapped.ale = proxy.original
        if env is not None:
            env.close()
        writer.close()
        input_hook.remove()

    if recorded < 30 * FPS:
        raise RuntimeError("Episode ended before the requested minimum clip duration")
    if weight_digest(actor) != initial_weights:
        raise RuntimeError("Actor parameters changed")
    for path, digest in before_hashes.items():
        if file_digest(path) != digest:
            raise RuntimeError(f"Protected file hash changed: {path}")
    verification = validate_video(temporary, recorded)
    temporary.rename(OUTPUT)
    print(json.dumps(dict(
        output=str(OUTPUT), checkpoint=str(CHECKPOINT), checkpoint_sha256=EXPECTED_HASH,
        seed=options.seed, seconds=recorded / FPS, actor_actions=dict(actions),
        stochastic_frameskip_counts=dict(skip_counts), scoring_rewards=rewards,
        episode_ended=done, simulator_frames_observed=proxy.frames,
        raw_rgb_stream_sha256=rgb_hash.hexdigest(), verification=verification,
        checkpoint_and_actor_hashes_verified=True,
    ), indent=2), flush=True)


if __name__ == "__main__":
    main()
