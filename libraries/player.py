import cv2

import libraries.common as common

def main(video_name):

    video = cv2.VideoCapture(common.VIDEOS_DIR + video_name)
    
    play_video(video)
    
    
def play_video(video):
    if type(video) != cv2.VideoCapture:
        print('video must be a cv2.VideoCapture object')
        return
    
    if (video.isOpened() is False):
        print('Error while trying to read video. Plese check again...')

    # Read until video is completed 
    while(video.isOpened()): 
        
    # Capture frame-by-frame 
        ret, frame = video.read() 
        if ret is True: 
        # Display the resulting frame 
            cv2.imshow('PRESS \'q\' TO CLOSE', frame) 
            
        # Press Q on keyboard to exit 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
    
    # Break the loop 
        else: 
            break
    
    # When everything done, release 
    # the video capture object 
    video.release() 
    
    # Closes all the frames 
    cv2.destroyAllWindows() 