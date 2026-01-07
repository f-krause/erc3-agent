import textwrap
from agent import run_agent
from erc3 import ERC3
from dotenv import load_dotenv

load_dotenv()
core = ERC3()
MODEL_ID = "gemini-2.5-flash"


def main():
    # Start session with metadata
    res = core.start_session(
        benchmark="erc3-dev",
        workspace="dev",
        name=f"given demo",
        architecture="given demo architecture",
        # can also set to compete_budget, compete_speed and/or compete_local
        flags=["compete_accuracy"]
    )

    status = core.session_status(res.session_id)
    print(f"Session has {len(status.tasks)} tasks")

    for task in status.tasks:
        print("=" * 40)
        print(f"Starting Task: {task.task_id} ({task.spec_id}): {task.task_text}")
        # start the task
        core.start_task(task)
        try:
            run_agent(MODEL_ID, core, task)
        except Exception as e:
            print(e)
        result = core.complete_task(task)
        if result.eval:
            explain = textwrap.indent(result.eval.logs, "  ")
            print(f"\nSCORE: {result.eval.score}\n{explain}\n")

    core.submit_session(res.session_id)


if __name__ == "__main__":
    main()
