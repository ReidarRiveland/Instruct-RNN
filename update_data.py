from instructRNN.data_loaders.dataset import build_training_data, TASK_LIST
from instructRNN.instructions.instructions_script import save_instruct_dicts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--tasks', default=TASK_LIST, nargs='*', help='list of tasks to remake data for')

    args = parser.parse_args()
    print('SAVING INSTRUCTIONS')
    save_instruct_dicts(args.path)
    for task in args.tasks: 
        build_training_data(args.path, task)
