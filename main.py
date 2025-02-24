from YOLO.trackers import ballTrackerClass
from utils import save_video, read_video
from trackers import playerTrackerClass
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import cv2

def main():
    video_frames = read_video('C:\\Users\\alber\\Desktop\\MyYolo\\YOLO\\input_video\\input_video.mp4')

    PlayerTracker = playerTrackerClass('models/NBA.pt')
    BallTracker = ballTrackerClass('models/ball_detector_model.pt')

    playersTracks = PlayerTracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='C:\\Users\\alber\\Desktop\\MyYolo\\YOLO\\stubs\\player_track_stubs.pkl')
    ballTracks = BallTracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='C:\\Users\\alber\\Desktop\\MyYolo\\YOLO\\stubs\\ball_track_stubs.pkl')

    ballTracks = BallTracker.interpolate_ball_positions(ballTracks)

    #Creating image for team assigning
    for player_id, player in playersTracks[0].items():
        bbox = player["bbox"]
        print(bbox)
        frame = video_frames[0]
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        cv2.imwrite('output_videos/cropped_image.jpg', cropped_image)
        cv2.imshow('cropped_image', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], playersTracks[0])

    for frame_num, player_track in enumerate(playersTracks):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            playersTracks[frame_num][player_id]['team'] = team
            playersTracks[frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(playersTracks):
        ball_bbox = ballTracks[frame_num]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            playersTracks[frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(playersTracks[frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])

    output_video_frames = PlayerTracker.draw_annotations(video_frames, playersTracks)
    output_video_frames = BallTracker.draw_annotations(output_video_frames, ballTracks)

    save_video(output_video_frames, 'C:\\Users\\alber\\Desktop\\MyYolo\\YOLO\\output_videos\\output_video1.avi')
if __name__ == '__main__':
    main()