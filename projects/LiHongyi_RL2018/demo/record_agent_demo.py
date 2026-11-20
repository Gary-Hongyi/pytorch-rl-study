"""Record a real, fixed-seed Double DQN rollout with simulator-frame timing.

Pong recording is pending verification of the missing historical environment,
Actor definition and preprocessing imports used by modern_pg_improved.
"""

import argparse
import ast
import gc
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import gymnasium as gym
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modern_double_dqn"))
from agent_dqn import Agent_DQN
from environment import Environment, make_breakout_env
from q_network import QNetwork

CHECKPOINT = ROOT / "modern_double_dqn/checkpoints/double_dqn_cpu_01_latest_1750000.pt"
OUTPUT = ROOT / "videos/breakout_double_dqn_latest.mp4"
EXPECTED_SHA256 = "44358f250f68ba43dd2c6de34e9cc6e38c166e93c01408bd17d5559a4132e324"
EPSILON = 0.025
FPS = 60
SIZE = (1280, 720)


def file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def weight_digest(model):
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def verify_shared_protocol():
    for filename in ("environment.py", "atari_wrapper.py", "q_network.py"):
        baseline = ROOT / "modern_dqn" / filename
        double = ROOT / "modern_double_dqn" / filename
        if baseline.read_bytes() != double.read_bytes():
            raise RuntimeError(f"Shared evaluation implementation differs: {filename}")
    methods = []
    for directory in ("modern_dqn", "modern_double_dqn"):
        tree = ast.parse((ROOT / directory / "agent_dqn.py").read_text(encoding="utf-8-sig"))
        selected = {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in ("make_action", "init_game_setting", "_observation_to_tensor")
        }
        methods.append(selected)
    if len(methods[0]) != 3 or methods[0] != methods[1]:
        raise RuntimeError("Inference methods differ from the multiseed evaluation implementation")


def make_facade(render_mode):
    # Reuse the original factory and the original four-value interface. The
    # factory's existing render_mode option is the only extra environment option.
    env = Environment.__new__(Environment)
    env.env = make_breakout_env(test=True, seed=42, render_mode=render_mode)
    env.action_space = env.env.action_space
    env.observation_space = env.env.observation_space
    env._pending_seed = None
    env.last_reset_info = {}
    return env


def make_agent(env, network, seed):
    agent = Agent_DQN.__new__(Agent_DQN)
    agent.env = env
    agent.action_dim = env.action_space.n
    agent.device = torch.device("cpu")
    agent.q_network = network
    agent.epsilon = EPSILON
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    return agent


class CaptureSimulatorFrames(gym.Wrapper):
    """Observe each existing one-frame ALE step below the skip/pooling wrapper."""

    def __init__(self, env, consume):
        super().__init__(env)
        self.consume = consume
        self.frames = 0

    def step(self, action):
        before = self.unwrapped.ale.getFrameNumber()
        result = self.env.step(action)
        elapsed = self.unwrapped.ale.getFrameNumber() - before
        if elapsed != 1:
            raise RuntimeError(f"Expected one simulator frame per base step, found {elapsed}")
        rgb = self.env.render()
        if rgb.shape != (210, 160, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f"Unexpected raw RGB frame: {rgb.shape}, {rgb.dtype}")
        self.frames += 1
        self.consume(rgb)
        return result


def attach_capture(env, consume):
    parent = env.env
    while parent.env is not parent.unwrapped:
        parent = parent.env
    capture = CaptureSimulatorFrames(parent.env, consume)
    parent.env = capture
    return capture


def verify_rendering_equivalence(network, seed):
    """Compare 64 decisions with/without capture, using identical RNG resets."""
    traces = []
    for rendered in (False, True):
        env = make_facade("rgb_array" if rendered else None)
        captured = []
        if rendered:
            attach_capture(env, lambda frame: captured.append(float(frame.std())))
        try:
            agent = make_agent(env, network, seed)
            state = env.reset()
            agent.init_game_setting()
            trace = []
            for _ in range(64):
                action = agent.make_action(state, test=True)
                state, reward, done, info = env.step(action)
                trace.append((action, reward, done, hashlib.sha256(state.tobytes()).hexdigest()))
                if done:
                    state = env.reset()
                    agent.init_game_setting()
            traces.append(trace)
            if rendered and (not captured or min(captured) <= 0):
                raise RuntimeError("RGB rendering produced empty frames")
        finally:
            env.close()
    if traces[0] != traces[1]:
        raise RuntimeError("Rendering/capture changed the evaluation trajectory")


def verify_video(path, expected_frames):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Cannot open the encoded MP4")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    previous = None
    changed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            game = frame[45:675, 400:880]
            if float(game.std()) < 1:
                raise RuntimeError(f"Black/corrupt game frame at index {count}")
            if previous is not None and np.any(game != previous):
                changed += 1
            previous = game
            count += 1
    finally:
        capture.release()
    if count != expected_frames or abs(fps - FPS) > 0.01 or (width, height) != SIZE:
        raise RuntimeError(f"Video metadata mismatch: {count=}, {fps=}, {width=}, {height=}")
    if changed < count // 4:
        raise RuntimeError("Too few changing game frames")
    return dict(decoded_frames=count, fps=fps, dimensions=[width, height], changed_frames=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11037)
    parser.add_argument("--seconds", type=int, default=60)
    options = parser.parse_args()
    if not 30 <= options.seconds <= 60:
        parser.error("Choose a duration between 30 and 60 seconds")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    temporary = OUTPUT.with_name(OUTPUT.stem + ".partial.mp4")
    if temporary.exists():
        raise FileExistsError(temporary)
    verify_shared_protocol()
    original_hash = file_digest(CHECKPOINT)
    if original_hash != EXPECTED_SHA256:
        raise RuntimeError("Checkpoint hash differs from the archived multiseed evaluation")
    torch.set_num_threads(1)
    # Trusted local project checkpoint contains NumPy RNG / replay objects.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(checkpoint["total_steps"]) != 1_750_000:
        raise RuntimeError("Incorrect checkpoint training node")
    network = QNetwork(action_dim=4)
    network.load_state_dict(checkpoint["online"], strict=True)
    network.eval().requires_grad_(False)
    del checkpoint
    gc.collect()
    original_weights = weight_digest(network)

    OUTPUT.parent.mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 25)
    canvas = Image.new("RGB", SIZE, (14, 27, 45))
    title = f"Double DQN   /   1,750,000 steps   /   Eval seed {options.seed}   /   epsilon 0.025"
    draw = ImageDraw.Draw(canvas)
    bounds = draw.textbbox((0, 0), title, font=font)
    draw.text(((SIZE[0] - bounds[2]) // 2, 7), title, fill=(235, 242, 248), font=font)
    frame_limit = options.seconds * FPS
    recorded = 0
    raw_hasher = hashlib.sha256()
    writer = imageio_ffmpeg.write_frames(
        str(temporary), SIZE, fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
        output_params=["-preset", "fast", "-crf", "18", "-movflags", "+faststart"],
    )

    def consume(rgb):
        nonlocal recorded
        if recorded >= frame_limit:
            return
        raw_hasher.update(rgb.tobytes())
        game = Image.fromarray(rgb).resize((480, 630), Image.Resampling.NEAREST)
        canvas.paste(game, (400, 45))
        writer.send(np.asarray(canvas))
        recorded += 1

    env = None
    try:
        with torch.inference_mode():
            verify_rendering_equivalence(network, options.seed)
            print("64-decision rendering equivalence check PASS", flush=True)
            env = make_facade("rgb_array")
            capture = attach_capture(env, consume)
            agent = make_agent(env, network, options.seed)
            writer.send(None)
            state = env.reset()
            agent.init_game_setting()
            actions = Counter()
            decision_reward = 0.0
            life_boundaries = 0
            real_game_over = False
            while recorded < frame_limit:
                action = agent.make_action(state, test=True)
                actions[action] += 1
                state, reward, done, info = env.step(action)
                decision_reward += float(reward)
                if done:
                    life_boundaries += 1
                    if info.get("real_done", False):
                        real_game_over = True
                        break
                    state = env.reset()
                    agent.init_game_setting()
    finally:
        if env is not None:
            env.close()
        writer.close()

    if file_digest(CHECKPOINT) != original_hash or weight_digest(network) != original_weights:
        raise RuntimeError("Checkpoint or inference weights changed")
    verification = verify_video(temporary, recorded)
    temporary.rename(OUTPUT)
    print(json.dumps(dict(
        output=str(OUTPUT), checkpoint=str(CHECKPOINT), checkpoint_sha256=original_hash,
        eval_seed=options.seed, seconds=recorded / FPS, actions=dict(actions),
        life_boundaries=life_boundaries, real_game_over=real_game_over,
        decision_reward_in_clip=decision_reward, simulator_frames_observed=capture.frames,
        raw_rgb_stream_sha256=raw_hasher.hexdigest(), verification=verification,
        weights_unchanged=True,
    ), indent=2), flush=True)


if __name__ == "__main__":
    main()
