import os
import zipfile
import shutil

OutputFolder = 'cleaned_data'

if not os.path.exists(OutputFolder):
    os.mkdir(OutputFolder)

ImageList = os.listdir(OutputFolder)
ImageList = [img for img in ImageList if len(img)>1] # filename string len >1
print('{} files in folder.'.format(len(ImageList)))

WordList = list(set([w.split('_')[0] for w in ImageList]))

def move_chars_into_dir():
    for w in WordList:
        try:
            by_char_dir = os.path.join(OutputFolder, w)
            if not os.path.exists(by_char_dir):
                os.mkdir(by_char_dir) # Create the new word folder in OutputPath.
            MoveList = [img for img in ImageList if w in img]

        except Exception as e:
            #print(e)
            MoveList = [ img for img in ImageList if w in img ]

        finally:
            for img in MoveList:
                old_path = OutputFolder + '/' + img
                new_path = OutputFolder + '/' + w + '/' + img
                shutil.move( old_path, new_path )

move_chars_into_dir()
