import os
import classify

def run():
    
    # get dataset's names from 'datasets' folder
    subfolders = [os.path.split(f.path)[-1] for f in os.scandir('datasets') if f.is_dir()]

    # looping clasification
    for name in subfolders:
        try:
            classify.run(name)
        except:
            print(f'Error is found, the {name} dataset is unable to process')

if __name__ == "__main__":
    run()