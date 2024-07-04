from pathlib import Path

def make_run_name(**kwargs):
    run_name = ""
    for k, v in kwargs.items():
        run_name += f"{k}-{v}_"
    run_name = run_name[:-1] # Remove trailing "_"
    return run_name

def _make_run_path(output_dir, run_name):
    return Path(output_dir) / "prediction" / run_name
    
def make_run_path(output_dir, **kwargs):
    return _make_run_path(
        output_dir, make_run_name(**kwargs)
    )