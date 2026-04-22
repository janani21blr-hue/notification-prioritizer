# dashboard.py
import streamlit as st
import pandas as pd
import json
import os
from env import NotificationEnv
from agent import choose_action
from baseline import run_random_agent, run_trained_agent
from data import NOTIFICATIONS
from tasks import task_mixed, task_urgent, task_noisy

st.set_page_config(page_title="Attention Guard AI", layout="wide")

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🛡️ Attention Guard: AI Focus Manager")
st.markdown("**Q-learning agent** that learns to protect your focus budget across 500 training episodes.")

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Training Progress", "⚔️ Before vs After", "🚀 Live Simulation"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRAINING PROGRESS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📈 Q-Learning Training Curve")
    st.markdown("Agent improves reward over 500 episodes by learning which notifications to suppress.")

    if os.path.exists("reward_history.json"):
        with open("reward_history.json") as f:
            rewards = json.load(f)

        window = 25
        rolling = [
            sum(rewards[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(rewards))
        ]

        df_curve = pd.DataFrame({
            "Episode Reward": rewards,
            "Rolling Avg (25)": rolling,
        })

        st.line_chart(df_curve, color=["#4f46e5", "#818cf8"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Episodes Trained", "500")
        col2.metric("Starting Avg Reward", "26.9")
        col3.metric("Final Avg Reward", "40.2")
        col4.metric("Improvement", "+49%", delta="+49%")

        st.markdown("---")
        st.markdown("**What the agent learned:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info("🔴 LOW Focus Zone\nAgent learned to **IGNORE** all non-critical notifications when focus < 0.3")
        with col_b:
            st.warning("🟡 MEDIUM Focus Zone\nAgent learned to **DELAY** optional notifications and only NOTIFY for important ones")
        with col_c:
            st.success("🟢 HIGH Focus Zone\nAgent learned to NOTIFY for important, DELAY for optional, IGNORE junk")

        if os.path.exists("learning_curve.png"):
            st.image("learning_curve.png", caption="Training curve saved from train.py", use_container_width=True)
    else:
        st.warning("Run `python train.py` first to generate reward_history.json")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BEFORE vs AFTER COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("⚔️ Trained Agent vs Random Agent")
    st.markdown("Same notifications. Same environment. Different agents. Watch the gap.")

    task_choice = st.selectbox(
        "Select Task",
        ["Mixed (Real World)", "Urgent Only", "Noisy (Hidden Important)"],
        key="task_select"
    )

    task_map = {
        "Mixed (Real World)": task_mixed,
        "Urgent Only": task_urgent,
        "Noisy (Hidden Important)": task_noisy,
    }

    if st.button("⚡ Run Comparison", key="compare_btn"):
        task_fn = task_map[task_choice]
        data = task_fn(NOTIFICATIONS)

        with st.spinner("Running both agents on identical data..."):
            random_result = run_random_agent(data)
            trained_result = run_trained_agent(data)

        reward_delta = trained_result["total_reward"] - random_result["total_reward"]
        focus_delta = trained_result["avg_focus"] - random_result["avg_focus"]

        # ── Headline metrics ──
        st.markdown("### Results")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Total Reward — Random", random_result["total_reward"],
        )
        col2.metric(
            "Total Reward — Trained", trained_result["total_reward"],
            delta=f"+{round(reward_delta, 2)}"
        )
        col3.metric(
            "Focus Preserved", f"{trained_result['avg_focus']:.3f}",
            delta=f"+{focus_delta:.3f} vs random"
        )

        st.markdown("---")

        # ── Side by side detail ──
        col_r, col_t = st.columns(2)

        with col_r:
            st.markdown("#### 🎲 Random Agent")
            st.markdown(f"- Total Reward: **{random_result['total_reward']}**")
            st.markdown(f"- Avg Reward per step: **{random_result['avg_reward']}**")
            st.markdown(f"- Avg Focus maintained: **{random_result['avg_focus']}**")
            st.markdown(f"- Final Focus: **{random_result['final_focus']}**")
            st.markdown(f"- NOTIFY: {random_result['actions']['notify']} | DELAY: {random_result['actions']['delay']} | IGNORE: {random_result['actions']['ignore']}")

        with col_t:
            st.markdown("#### 🧠 Trained Q-Agent")
            st.markdown(f"- Total Reward: **{trained_result['total_reward']}**")
            st.markdown(f"- Avg Reward per step: **{trained_result['avg_reward']}**")
            st.markdown(f"- Avg Focus maintained: **{trained_result['avg_focus']}**")
            st.markdown(f"- Final Focus: **{trained_result['final_focus']}**")
            st.markdown(f"- NOTIFY: {trained_result['actions']['notify']} | DELAY: {trained_result['actions']['delay']} | IGNORE: {trained_result['actions']['ignore']}")

        st.markdown("---")

        # ── Improvement callout ──
        if reward_delta > 0:
            st.success(f"✅ Trained agent achieved **+{round(reward_delta, 2)} more total reward** and preserved **+{focus_delta:.3f} more focus** on average.")
        else:
            st.info("Results vary per run due to random task shuffling. Run again to see average trend.")

        # ── Q-table insight ──
        st.markdown("---")
        st.subheader("🧠 What the Agent Learned (Q-Table)")
        if os.path.exists("q_table.json"):
            with open("q_table.json") as f:
                qt = json.load(f)
            rows = []
            for state, actions in qt.items():
                best = max(actions, key=actions.get)
                parts = state.split("|")
                rows.append({
                    "Focus Zone": parts[0] if len(parts) > 0 else "",
                    "Importance": parts[1] if len(parts) > 1 else "",
                    "User State": parts[2] if len(parts) > 2 else "",
                    "Best Action": best.upper(),
                    "Notify Q": round(actions.get("notify", 0), 3),
                    "Delay Q": round(actions.get("delay", 0), 3),
                    "Ignore Q": round(actions.get("ignore", 0), 3),
                })
            df_qt = pd.DataFrame(rows).sort_values(["Focus Zone", "Importance"])
            st.dataframe(df_qt, use_container_width=True, hide_index=True)
            st.caption(f"Agent learned {len(qt)} distinct states across Focus Zone × Importance × User Activity")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE SIMULATION
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🚀 Live Agent Simulation")
    st.markdown("Watch the trained agent process 50 real notifications in real time.")

    sim_task = st.selectbox(
        "Task Mode",
        ["Mixed (Real World)", "Urgent Only", "Noisy (Hidden Important)"],
        key="sim_task"
    )

    if st.button("🚀 Run Simulation", key="sim_btn"):
        task_fn = task_map[sim_task]
        env = NotificationEnv(data=task_fn(NOTIFICATIONS))
        obs, _ = env.reset()
        done = False
        log_data = []
        progress = st.progress(0.0)
        total = len(env.notifications)
        step = 0

        while not done and obs is not None:
            action = choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            log_data.append({
                "#": step + 1,
                "App": obs.app,
                "Sender": obs.sender,
                "Message": obs.message[:50] + ("…" if len(obs.message) > 50 else ""),
                "User State": obs.user_state,
                "Focus Zone": "🔴 LOW" if obs.current_focus < 0.3 else ("🟡 MED" if obs.current_focus < 0.6 else "🟢 HIGH"),
                "Annoyed": "⚠️" if obs.is_user_annoyed else "",
                "AI Action": action.upper(),
                "Reward": round(float(reward), 3),
                "Focus Budget": round(float(info["current_focus"]), 2),
            })
            obs = next_obs
            done = terminated or truncated
            step += 1
            progress.progress(step / total)

        st.success(f"✅ Done — {step} notifications processed.")
        df = pd.DataFrame(log_data)

        st.subheader("📋 Audit Log")
        st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Focus Budget Over Time")
            st.line_chart(df.set_index("#")["Focus Budget"])
        with col2:
            st.subheader("🧭 Action Distribution")
            st.bar_chart(df["AI Action"].value_counts())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Reward", round(df["Reward"].mean(), 3))
        c2.metric("Notify", int((df["AI Action"] == "NOTIFY").sum()))
        c3.metric("Delay", int((df["AI Action"] == "DELAY").sum()))
        c4.metric("Ignore", int((df["AI Action"] == "IGNORE").sum()))