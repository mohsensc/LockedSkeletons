# LockedSkeletons

Eye tracker that plays a skeleton banging a shield when you look away from the screen so you lock in instead of scrolling or falling asleep

## Setup Instructions

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python3 script.py
```

## Controls / Customization

- Edit lines 8-23 of script.py to adjust sensitivity of app, eye shape, gaze boundaries, etc. Save changes with Command-S before re-running
- Run `^C` to quit the application

## How It Works

1. Captures video from your webcam
2. Detects eyes using MediaPipe
3. Gets mad at you if:
   - you’re looking away (gaze not within allowed gaze box)
   - your eyes are closed for longer than a short grace period (to ignore normal blinks)
   - no face is detected
4. If you’re unfocused for longer than the threshold, it plays the skeleton banging on the shield
5. Restarts once you lock in again, opening a new skeleton shield window every time.

