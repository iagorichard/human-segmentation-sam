
## Example of usage:

### To process a directory containing a lot of videos:
python segment_image.py --supervideo_superpath /folder/to/directory/root/

### To process a unique "supervideo":
python segment_image.py --video_path /folder/to/directory/root/supervideo.mp4

### To process a unique image:
python segment_image.py --image_path /folder/to/directory/root/image.jpg

### To visualize a "supervideo" and its annotations:
python show_segmented.py --supervideo_path /folder/to/directory/root/supervideo.mp4 --json_path /folder/to/directory/root/annot_json.json
