import os
import shutil
from datetime import datetime, timedelta


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Get the list of all the directories and exclude the directories with
last_3_days = [(datetime.now() - timedelta(days=x)).strftime("%d-%m-%Y") for x in xrange(3)]

for photo_dir in ["Photos", "Faces"]:
    dir_contents = [
        os.path.join(BASE_PATH, photo_dir, x) for x in os.listdir(os.path.join(BASE_PATH, photo_dir)) if x not in last_3_days]
    for dir_content in dir_contents:
        shutil.rmtree(dir_content)
        print "Deleted", dir_content
