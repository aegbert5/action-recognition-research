# Import a library of functions called 'pygame'
import pygame
from math import pi
import numpy as np
import math

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Point3D:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
		
class Line3D():
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.start4D = np.array([self.start.x,self.start.y,self.start.z,1])
        self.end4D = np.array([self.end.x,self.end.y,self.end.z,1])

def loadOBJ(filename):
	
	vertices = []
	indices = []
	lines = []
	
	f = open(filename, "r")
	for line in f:
		t = str.split(line)
		if not t:
			continue
		if t[0] == "v":
			vertices.append(Point3D(float(t[1]),float(t[2]),float(t[3])))
			
		if t[0] == "f":
			for i in range(1,len(t) - 1):
				index1 = int(str.split(t[i],"/")[0])
				index2 = int(str.split(t[i+1],"/")[0])
				indices.append((index1,index2))
			
	f.close()
	
	#Add faces as lines
	for index_pair in indices:
		index1 = index_pair[0]
		index2 = index_pair[1]
		lines.append(Line3D(vertices[index1 - 1],vertices[index2 - 1]))
		
	#Find duplicates
	duplicates = []
	for i in range(len(lines)):
		for j in range(i+1, len(lines)):
			line1 = lines[i]
			line2 = lines[j]
			
			# Case 1 -> Starts match
			if line1.start.x == line2.start.x and line1.start.y == line2.start.y and line1.start.z == line2.start.z:
				if line1.end.x == line2.end.x and line1.end.y == line2.end.y and line1.end.z == line2.end.z:
					duplicates.append(j)
			# Case 2 -> Start matches end
			if line1.start.x == line2.end.x and line1.start.y == line2.end.y and line1.start.z == line2.end.z:
				if line1.end.x == line2.start.x and line1.end.y == line2.start.y and line1.end.z == line2.start.z:
					duplicates.append(j)
					
	duplicates = list(set(duplicates))
	duplicates.sort()
	duplicates = duplicates[::-1]
	
	#Remove duplicates
	for j in range(len(duplicates)):
		del lines[duplicates[j]]
	
	return lines

def loadSkeleton(action, frame):

    spine_base_x, spine_base_y, spine_base_z = action[:,frame,0]
    spine_mid_x, spine_mid_y, spine_mid_z = action[:,frame,1]
    neck_x, neck_y, neck_z = action[:,frame,2]
    head_x, head_y, head_z = action[:,frame,3]

    shoulder_left_x, shoulder_left_y, shoulder_left_z = action[:,frame,4]
    elbow_left_x, elbow_left_y, elbow_left_z = action[:,frame,5]
    wrist_left_x, wrist_left_y, wrist_left_z = action[:,frame,6]
    hand_left_x, hand_left_y, hand_left_z = action[:,frame,7]
    
    shoulder_right_x, shoulder_right_y, shoulder_right_z = action[:,frame,8]
    elbow_right_x, elbow_right_y, elbow_right_z = action[:,frame,9]
    wrist_right_x, wrist_right_y, wrist_right_z = action[:,frame,10]
    hand_right_x, hand_right_y, hand_right_z = action[:,frame,11]

    hip_left_x, hip_left_y, hip_left_z = action[:,frame,12]
    knee_left_x, knee_left_y, knee_left_z = action[:,frame,13]
    ankle_left_x, ankle_left_y, ankle_left_z = action[:,frame,14]
    foot_left_x, foot_left_y, foot_left_z = action[:,frame,15]
    
    hip_right_x, hip_right_y, hip_right_z = action[:,frame,16]
    knee_right_x, knee_right_y, knee_right_z = action[:,frame,17]
    ankle_right_x, ankle_right_y, ankle_right_z = action[:,frame,18]
    foot_right_x, foot_right_y, foot_right_z = action[:,frame,19]
    
    spine_shoulder_x, spine_shoulder_y, spine_shoulder_z = action[:,frame,20]

    hand_tip_left_x, hand_tip_left_y, hand_tip_left_z = action[:,frame,21]
    thumb_left_x, thumb_left_y, thumb_left_z = action[:,frame,22]
    
    hand_tip_right_x, hand_tip_right_y, hand_tip_right_z = action[:,frame,23]
    thumb_right_x, thumb_right_y, thumb_right_z = action[:,frame,24]

    skeleton = []
    
    #Spine
    skeleton.append(WHITE)
    skeleton.append(Line3D(Point3D(spine_base_x, spine_base_y, spine_base_z), Point3D(spine_mid_x, spine_mid_y, spine_mid_z)))
    skeleton.append(Line3D(Point3D(spine_shoulder_x, spine_shoulder_y, spine_shoulder_z), Point3D(spine_mid_x, spine_mid_y, spine_mid_z)))
    skeleton.append(Line3D(Point3D(neck_x, neck_y, neck_z), Point3D(spine_shoulder_x, spine_shoulder_y, spine_shoulder_z)))
    skeleton.append(Line3D(Point3D(neck_x, neck_y, neck_z), Point3D(head_x, head_y, head_z)))
    
    #Left Leg
    skeleton.append(RED)
    skeleton.append(Line3D(Point3D(hip_left_x, hip_left_y, hip_left_z),Point3D(spine_base_x, spine_base_y, spine_base_z)))
    skeleton.append(Line3D(Point3D(knee_left_x, knee_left_y, knee_left_z),Point3D(hip_left_x, hip_left_y, hip_left_z)))
    skeleton.append(Line3D(Point3D(ankle_left_x, ankle_left_y, ankle_left_z),Point3D(knee_left_x, knee_left_y, knee_left_z)))
    skeleton.append(Line3D(Point3D(foot_left_x, foot_left_y, foot_left_z),Point3D(ankle_left_x, ankle_left_y, ankle_left_z)))

    #Right Leg
    skeleton.append(GREEN)
    skeleton.append(Line3D(Point3D(hip_right_x, hip_right_y, hip_right_z),Point3D(spine_base_x, spine_base_y, spine_base_z)))
    skeleton.append(Line3D(Point3D(knee_right_x, knee_right_y, knee_right_z),Point3D(hip_right_x, hip_right_y, hip_right_z)))
    skeleton.append(Line3D(Point3D(ankle_right_x, ankle_right_y, ankle_right_z),Point3D(knee_right_x, knee_right_y, knee_right_z)))
    skeleton.append(Line3D(Point3D(foot_right_x, foot_right_y, foot_right_z),Point3D(ankle_right_x, ankle_right_y, ankle_right_z)))
    
    #Left Arm
    skeleton.append(RED)
    skeleton.append(Line3D(Point3D(shoulder_left_x, shoulder_left_y, shoulder_left_z),Point3D(spine_shoulder_x, spine_shoulder_y, spine_shoulder_z)))
    skeleton.append(Line3D(Point3D(elbow_left_x, elbow_left_y, elbow_left_z),Point3D(shoulder_left_x, shoulder_left_y, shoulder_left_z)))
    skeleton.append(Line3D(Point3D(wrist_left_x, wrist_left_y, wrist_left_z),Point3D(elbow_left_x, elbow_left_y, elbow_left_z)))
    skeleton.append(Line3D(Point3D(hand_left_x, hand_left_y, hand_left_z),Point3D(wrist_left_x, wrist_left_y, wrist_left_z)))
    skeleton.append(Line3D(Point3D(hand_tip_left_x, hand_tip_left_y, hand_tip_left_z),Point3D(hand_left_x, hand_left_y, hand_left_z)))

    skeleton.append(Line3D(Point3D(thumb_left_x, thumb_left_y, thumb_left_z),Point3D(hand_left_x, hand_left_y, hand_left_z)))
    
    #Right Arm
    skeleton.append(GREEN)
    skeleton.append(Line3D(Point3D(shoulder_right_x, shoulder_right_y, shoulder_right_z),Point3D(spine_shoulder_x, spine_shoulder_y, spine_shoulder_z)))
    skeleton.append(Line3D(Point3D(elbow_right_x, elbow_right_y, elbow_right_z),Point3D(shoulder_right_x, shoulder_right_y, shoulder_right_z)))
    skeleton.append(Line3D(Point3D(wrist_right_x, wrist_right_y, wrist_right_z),Point3D(elbow_right_x, elbow_right_y, elbow_right_z)))
    skeleton.append(Line3D(Point3D(hand_right_x, hand_right_y, hand_right_z),Point3D(wrist_right_x, wrist_right_y, wrist_right_z)))
    skeleton.append(Line3D(Point3D(hand_tip_right_x, hand_tip_right_y, hand_tip_right_z),Point3D(hand_right_x, hand_right_y, hand_right_z)))
    
    skeleton.append(Line3D(Point3D(thumb_right_x, thumb_right_y, thumb_right_z),Point3D(hand_right_x, hand_right_y, hand_right_z)))
    return skeleton

def loadHouse():
    house = []
    #Floor
    house.append(RED)
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(5, 0, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 0, 5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(-5, 0, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 0, -5)))
    #Ceiling
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 5, -5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(5, 5, 5), Point3D(-5, 5, 5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(-5, 5, -5)))
    #Walls
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(-5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 5, 5)))
    #Door
    house.append(Line3D(Point3D(-1, 0, 5), Point3D(-1, 3, 5)))
    house.append(Line3D(Point3D(-1, 3, 5), Point3D(1, 3, 5)))
    house.append(Line3D(Point3D(1, 3, 5), Point3D(1, 0, 5)))
    #Roof
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(0, 8, -5)))
    house.append(Line3D(Point3D(0, 8, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(0, 8, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(0, 8, -5)))
	
    return house

def loadCar():
    car = []
    #Front Side
    car.append(GREEN)
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-2, 3, 2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(2, 3, 2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(3, 2, 2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 1, 2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(-3, 1, 2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 2, 2)))

    #Back Side
    car.append(Line3D(Point3D(-3, 2, -2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(-2, 3, -2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, -2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 2, -2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(3, 1, -2), Point3D(-3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, -2), Point3D(-3, 2, -2)))
    
    #Connectors
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-3, 2, -2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 1, -2)))

    return car

def loadTire():
    tire = []
    #Front Side
    tire.append(BLUE)
    tire.append(Line3D(Point3D(-1, .5, .5), Point3D(-.5, 1, .5)))
    tire.append(Line3D(Point3D(-.5, 1, .5), Point3D(.5, 1, .5)))
    tire.append(Line3D(Point3D(.5, 1, .5), Point3D(1, .5, .5)))
    tire.append(Line3D(Point3D(1, .5, .5), Point3D(1, -.5, .5)))
    tire.append(Line3D(Point3D(1, -.5, .5), Point3D(.5, -1, .5)))
    tire.append(Line3D(Point3D(.5, -1, .5), Point3D(-.5, -1, .5)))
    tire.append(Line3D(Point3D(-.5, -1, .5), Point3D(-1, -.5, .5)))
    tire.append(Line3D(Point3D(-1, -.5, .5), Point3D(-1, .5, .5)))

    #Back Side
    tire.append(Line3D(Point3D(-1, .5, -.5), Point3D(-.5, 1, -.5)))
    tire.append(Line3D(Point3D(-.5, 1, -.5), Point3D(.5, 1, -.5)))
    tire.append(Line3D(Point3D(.5, 1, -.5), Point3D(1, .5, -.5)))
    tire.append(Line3D(Point3D(1, .5, -.5), Point3D(1, -.5, -.5)))
    tire.append(Line3D(Point3D(1, -.5, -.5), Point3D(.5, -1, -.5)))
    tire.append(Line3D(Point3D(.5, -1, -.5), Point3D(-.5, -1, -.5)))
    tire.append(Line3D(Point3D(-.5, -1, -.5), Point3D(-1, -.5, -.5)))
    tire.append(Line3D(Point3D(-1, -.5, -.5), Point3D(-1, .5, -.5)))

    #Connectors
    tire.append(Line3D(Point3D(-1, .5, .5), Point3D(-1, .5, -.5)))
    tire.append(Line3D(Point3D(-.5, 1, .5), Point3D(-.5, 1, -.5)))
    tire.append(Line3D(Point3D(.5, 1, .5), Point3D(.5, 1, -.5)))
    tire.append(Line3D(Point3D(1, .5, .5), Point3D(1, .5, -.5)))
    tire.append(Line3D(Point3D(1, -.5, .5), Point3D(1, -.5, -.5)))
    tire.append(Line3D(Point3D(.5, -1, .5), Point3D(.5, -1, -.5)))
    tire.append(Line3D(Point3D(-.5, -1, .5), Point3D(-.5, -1, -.5)))
    tire.append(Line3D(Point3D(-1, -.5, .5), Point3D(-1, -.5, -.5)))
    
    return tire
    
def append_tire_to_car(tire, xpos, ypos, zpos, tire_rotation):
    rotation = get_rotation_matrix(0,0,tire_rotation)
    translation = get_translation_matrix(xpos,ypos,zpos)
    final_matrix = np.matmul(translation, rotation)

    # Apply the transformation to the tire
    for idx, line in enumerate(tire):
        if (line == RED or line == GREEN or line == BLUE):
            tire[idx] = line
        else:
            start4D = np.matmul(final_matrix, np.asarray(line.start4D, dtype=np.float32))
            end4D = np.matmul(final_matrix, np.asarray(line.end4D, dtype=np.float32))
            tire[idx] = Line3D(Point3D(start4D[0], start4D[1], start4D[2]), Point3D(end4D[0], end4D[1], end4D[2]))

    return tire

def loadTransformedCar(tire_rotation):
    lines = loadCar()

    tire1 = append_tire_to_car(loadTire(), 2,0,2, tire_rotation)
    tire2 = append_tire_to_car(loadTire(), 2,0,-2, tire_rotation)
    tire3 = append_tire_to_car(loadTire(), -2,0,2, tire_rotation)
    tire4 = append_tire_to_car(loadTire(), -2,0,-2, tire_rotation)
    
    result = lines + tire1 + tire2 + tire3 + tire4
    return result
    

def get_identity_matrix():
    matrix = np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
    return matrix

def get_translation_matrix(x,y,z):
    matrix = np.array([[1,0,0,x],
                      [0,1,0,y],
                      [0,0,1,z],
                      [0,0,0,1]])
    return matrix

def get_scale_matrix(x,y,z):
    matrix = np.array([[x,0,0,0],
                      [0,y,0,0],
                      [0,0,z,0],
                      [0,0,0,1]])
    return matrix

def get_rotation_matrix(rot_x,rot_y,rot_z):
    cos_value = math.cos(math.radians(rot_x))
    sin_value = math.sin(math.radians(rot_x))
    rotation_x = np.array([[1,0,0,0],
                          [0,cos_value,-sin_value,0],
                          [0,sin_value,cos_value,0],
                          [0,0,0,1]])

    cos_value = math.cos(math.radians(rot_y))
    sin_value = math.sin(math.radians(rot_y))
    rotation_y = np.array([[cos_value,0,-sin_value,0],
                          [0,1,0,0],
                          [sin_value,0,cos_value,0],
                          [0,0,0,1]])

    cos_value = math.cos(math.radians(rot_z))
    sin_value = math.sin(math.radians(rot_z))
    rotation_z = np.array([[cos_value,-sin_value,0,0],
                          [sin_value,cos_value,0,0],
                          [0,0,1,0],
                          [0,0,0,1]])

    result = np.matmul(np.matmul(rotation_x,rotation_y),rotation_z)
    return result

def get_object_to_world_matrix(scale_x,scale_y,scale_z, rot_x,rot_y,rot_z, trans_x,trans_y,trans_z):
    scale = get_scale_matrix(scale_x,scale_y,scale_z)
    rotation = get_rotation_matrix(rot_x,rot_y,rot_z)
    translation = get_translation_matrix(trans_x,trans_y,trans_z)
    result = np.matmul(np.matmul(translation,rotation),scale)
    return result

def get_world_to_camera_matrix(cam_x, cam_y, cam_z, rot_x, rot_y, rot_z):
    translation = get_translation_matrix(-cam_x, -cam_y, -cam_z)
    rotation = get_rotation_matrix(rot_x,rot_y,rot_z)
    result = np.matmul(rotation,translation)

    #result = np.ndarray([xi.to4D() for xi in lines_3d], dtype=object)
    return result

def get_clip_matrix():
    fov_y = math.pi/3
    aspect_ratio = 1/1

    zoom_y = 1/math.tan(fov_y/2)
    zoom_x = zoom_y

    far_plane = 100
    near_plane = 10

    value1 = (far_plane + near_plane) / (far_plane - near_plane)
    value2 = (-2 * near_plane * far_plane) / (far_plane - near_plane)

    matrix = np.array([[zoom_x,0,0,0],
                      [0,zoom_y,0,0],
                      [0,0,value1,value2],
                      [0,0,1,0]])
    return matrix


def get_screen_matrix(size):
    width = size[0]
    height = size[1]
    matrix = np.array([[width/2,0,width/2],
                      [0,-height/2,height/2],
                      [0,0,1]])
    return matrix

def get_canonical_coordinates(coordinates):
    coordinates /= coordinates[3]
    return coordinates

# Applies the object to clip matrix and returns whether the line should be visible on the screen
def line_to_screen(line3d, obj_to_clip_matrix):
    clip_coordinates_start = np.matmul(obj_to_clip_matrix, line3d.start4D)
    clip_coordinates_end = np.matmul(obj_to_clip_matrix, line3d.end4D)
    
    clip_coordinates_start /= clip_coordinates_start[3]
    clip_coordinates_end /= clip_coordinates_end[3]

    # all of the tests pass unless otherwise found to be out of the viewing region
    test1 = True
    test2 = True
    test3 = True
    test4 = True
    test5 = True
    test6 = True

    if (clip_coordinates_start[0] < -1 and clip_coordinates_end[0] < -1):
      test1 = False
    if (clip_coordinates_start[0] > 1 and clip_coordinates_end[0] > 1):
      test2 = False
    if (clip_coordinates_start[1] < -1 and clip_coordinates_end[1] < -1):
      test3 = False
    if (clip_coordinates_start[1] > 1 and clip_coordinates_end[1] > 1):
      test4 = False

    # Near plane test will fail if either fails the test
    if (clip_coordinates_start[2] < -1 or clip_coordinates_end[2] < -1):
      test5 = False
    if (clip_coordinates_start[2] > 1 and clip_coordinates_end[2] > 1):
      test6 = False
                
    is_line_drawn = test1 and test2 and test3 and test4 and test5 and test6
    
    # Remove the z-value and keep the [x,y,1] vector 
    canonical_coordinates_start = np.delete(clip_coordinates_start, 2)
    canonical_coordinates_end = np.delete(clip_coordinates_end, 2)
    
    screen_coordinates_start = np.matmul(screen_matrix, canonical_coordinates_start)
    screen_coordinates_end = np.matmul(screen_matrix, canonical_coordinates_end)

    return (screen_coordinates_start, screen_coordinates_end, is_line_drawn)

# Camera variables
camera_x_start = 0
camera_y_start = 0
camera_z_start = -10

angle_x_start = 0
angle_y_start = 0
angle_z_start = 0

# Set the height and width of the screen
size = [512, 512]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Shape Drawing")
screen_matrix = get_screen_matrix(size)

# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)
         
class SkeletonVisualizer():

    def __init__(self,skeleton1,skeleton2):
        
        frame_animation_start = 0
        
        angle_x = angle_x_start
        angle_y = angle_y_start
        angle_z = angle_z_start
        
        camera_x = camera_x_start
        camera_y = camera_y_start
        camera_z = camera_z_start
        
        movement_sensitivity = 0.25
        turn_sensitivity = 0.25
        
        tire_rotation = 0
        
        
        clip_matrix = get_clip_matrix()
        
        # Initialize the game engine
        pygame.init()
         
        #Set needed variables
        done = False
        clock = pygame.time.Clock()
        start = Point(0.0,0.0)
        end = Point(0.0,0.0)
        static_object_list = [
                      #[loadHouse(), [1,1,1, 0,0,0, 20,0,-30]],
                      #[loadHouse(), [1,1,1, 0,0,0, 0,0,-30]],
                      #[loadHouse(), [1,1,1, 0,0,0, -20,0,-30]],
                      #[loadHouse(), [1,1,1, 0,-90,0, -30,0,0]],
                      #[loadHouse(), [1,1,1, 0,180,0, -20,0,30]],
                      #[loadHouse(), [1,1,1, 0,180,0, 0,0,30]],
                      #[loadHouse(), [1,1,1, 0,180,0, 20,0,30]]
                    ]
        
        #Loop until the user clicks the close button.
        while not done:
                    
                num_frames = pygame.time.get_ticks() - frame_animation_start
        
                animated_object_list = [
                                        [loadSkeleton(skeleton1, (num_frames//33) % 100), [10,10,10, 0,0,0, -10,0,10]],
                                        [loadSkeleton(skeleton2, (num_frames//33) % 100), [10,10,10, 0,0,0, 10,0,10]]
                                      ]
        
                object_list = static_object_list + animated_object_list
         
        	# This limits the while loop to a max of 100 times per second.
        	# Leave this out and we will use all CPU we can.
                clock.tick(100)
        
        	# Clear the screen and set the screen background
                screen.fill(BLACK)
        
        	#Controller Code#
        	#####################################################################
        
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: # If user clicked close
                        done=True
        			
                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_w]:
                    camera_z += math.cos(math.radians(-angle_y)) * movement_sensitivity
                    camera_x -= math.sin(math.radians(-angle_y)) * movement_sensitivity
                
                if pressed[pygame.K_s]:
                    camera_z -= math.cos(math.radians(-angle_y)) * movement_sensitivity
                    camera_x += math.sin(math.radians(-angle_y)) * movement_sensitivity
        
                if pressed[pygame.K_a]:
                    camera_z += math.cos(math.radians(-angle_y + 90)) * movement_sensitivity
                    camera_x -= math.sin(math.radians(-angle_y + 90)) * movement_sensitivity
                
                if pressed[pygame.K_d]:
                    camera_z -= math.cos(math.radians(-angle_y + 90)) * movement_sensitivity
                    camera_x += math.sin(math.radians(-angle_y + 90)) * movement_sensitivity
                
                if pressed[pygame.K_r]:
                    camera_y += movement_sensitivity
                
                if pressed[pygame.K_f]:
                    camera_y -= movement_sensitivity
                
                if pressed[pygame.K_e]:
                    angle_y += turn_sensitivity
                
                if pressed[pygame.K_q]:
                    angle_y -= turn_sensitivity
                
                if pressed[pygame.K_h]:
                    camera_x = camera_x_start
                    camera_y = camera_y_start
                    camera_z = camera_z_start
                        
                    angle_x = angle_x_start
                    angle_y = angle_y_start
                    angle_z = angle_z_start
        
                    frame_animation_start = pygame.time.get_ticks()
        	
                world_to_camera = get_world_to_camera_matrix(camera_x,camera_y,camera_z, angle_x,angle_y,angle_z)
                
                #Viewer Code#
        	#####################################################################
        
                for obj in object_list:
                        lines = obj[0]
                        location = obj[1]
        
                        scale_x = location[0]
                        scale_y = location[1]
                        scale_z = location[2]
                        
                        rot_x = location[3]
                        rot_y = location[4]
                        rot_z = location[5]
        
                        pos_x = location[6]
                        pos_y = location[7]
                        pos_z = location[8]
        
                        obj_to_world = get_object_to_world_matrix(scale_x,scale_y,scale_z, rot_x,rot_y,rot_z, pos_x,pos_y,pos_z)
                        obj_to_clip_matrix = np.matmul(clip_matrix, np.matmul(world_to_camera, obj_to_world))
                        
                        # Default Color is RED unless told otherwise
                        current_color = RED
                        for line in lines:
                            if (line == RED or line == BLUE or line == GREEN or line == WHITE):
                                current_color = line
                            else:
                                screen_line_start, screen_line_end, is_line_drawn = line_to_screen(line, obj_to_clip_matrix)
        
                                if (is_line_drawn):		
                                    pygame.draw.line(screen, current_color, (screen_line_start[0], screen_line_start[1]), (screen_line_end[0], screen_line_end[1]), 2)
        
        	# Go ahead and update the screen with what we've drawn.
        	# This MUST happen after all the other drawing commands.
                pygame.display.flip()
                #break
         
        # Be IDLE friendly
        pygame.quit()
