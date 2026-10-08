import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)             

sys.path.append(project_root)
sys.path.append(current_dir)

from paths import get_paths 


from tools.code_handling import code_handling   
from tools.git_actions import GitActions       
from tools.plot_progress import plot_progress
from tools.output_handeling import output_handeling

train_path = get_paths().get("train_path")
progress_path = get_paths().get("progress_path")

def get_tools():
    return {
        "code_handling": code_handling(train_path), 
        "output_handeling":output_handeling(),
        "git_actions": GitActions(train_path),    
        "plot_progress": plot_progress(progress_path)
    }
