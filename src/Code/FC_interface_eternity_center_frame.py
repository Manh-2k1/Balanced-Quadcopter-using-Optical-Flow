# import libraries
from operator import truediv
from pickle import TRUE
import time
import curses
from collections import deque
from itertools import cycle
from tkinter import Frame
from yamspy import MSPy
import tfli2c as tfl    #  Import `tfli2c` module v0.0.1
import sys
import threading
from ascii_convert import *
import cv2
import numpy as np
from ssc import ssc
import math

# Max periods for:
CTRL_LOOP_TIME = 1/100
SLOW_MSGS_LOOP_TIME = 1/5 # these messages take a lot of time slowing down the loop...

NO_OF_CYCLES_AVERAGE_GUI_TIME = 10

# open port
SERIAL_PORT = "/dev/ttyUSB0"

#   - - -  Set I2C Port and Address numbers  - - - - - - - -
I2CAddr = 0x10   # Device address in Hex, Decimal 16
I2CPort = 1      # I2C(4), /dev/i2c-4, GPIO 8/9, pins 24/21

if( tfl.begin( I2CAddr, I2CPort)):
    print( "Ready")
else:
    print( "Not ready")
    sys.exit()   #  quit the program if not ready

# Read Sensor
def LiDAR_reader():
    tfl.getData()      # Get tfl data
    return tfl.dist /100 # Get distance data from tfl

# PID_config for holding height mission
Kp = 0.2
Ki = 0.2
Kd = 10
PID_p = 0
PID_i = 0
PID_d = 0
PreviousError = 0
DesiredHeight = 1.2

def PID_calculation(ActualSignal):
    global PreviousError, PID_i
    Error = DesiredHeight - ActualSignal
    PID_p = Kp * Error
    dist_Error = Error - PreviousError
    PID_d = Kd*((Error - dist_Error)/CTRL_LOOP_TIME)
    if ((Error >=-3) and (Error <=3)):
        PID_i = PID_i + Ki*Error
    else:
        PID_i = 0
    PreviousError = Error
    return PID_p + PID_d + PID_i # PID total

# PID_config for Y-axis stability
Kp_pitch = 1.0 #0.205
Ki_pitch = 0.0 #0.0
Kd_pitch = 7.0  #0.0
PID_pitch_p = 0
PID_pitch_i = 0
PID_pitch_d = 0
PreviousError_Pitch = 0
DesiredY = 0

def PID_POS_pitch_calculation(Y_error):
    global Kp_pitch, Ki_pitch, Kd_pitch, PID_pitch_p, PID_pitch_i, PID_pitch_d, PreviousError_Pitch, DesiredY
    Pitch_Error = - Y_error
    PID_pitch_p = Kp_pitch * Y_error
    dist_Pitch_Error = Pitch_Error - PreviousError_Pitch
    PID_pitch_d = Kd_pitch*((Pitch_Error - PreviousError_Pitch)/CTRL_LOOP_TIME)
    if ((Pitch_Error >=-3) and (Pitch_Error <=3)):
        PID_pitch_i = PID_pitch_i + Ki_pitch*Pitch_Error
    else:
        PID_pitch_i = 0
    PreviousError_Pitch = Pitch_Error
    return PID_pitch_p + PID_pitch_i + PID_pitch_d # PID total

# PID_config for X-axis stability
Kp_roll = 1.0    #0.09
Ki_roll = 0.0    #0.0
Kd_roll = 2.0     #0.0532
PID_roll_p = 0
PID_roll_i = 0
PID_roll_d = 0
PreviousError_roll = 0
DesiredX = 0

def PID_POS_roll_calculation(X_error):
    global Kp_roll, Ki_roll, Kd_roll, PID_roll_p, PID_roll_i, PID_roll_d, PreviousError_roll, DesiredX
    Roll_Error = X_error
    PID_roll_p = Kp_roll * Roll_Error
    dist_Roll_Error = Roll_Error - PreviousError_roll
    PID_roll_d = Kd_roll*((Roll_Error - PreviousError_roll)/CTRL_LOOP_TIME)
    if ((Roll_Error >=-3) and (Roll_Error <=3)):
        PID_roll_i = PID_roll_i + Ki_roll*Roll_Error
    else:
        PID_roll_i = 0
    PreviousError_roll = Roll_Error
    return PID_roll_p + PID_roll_i + PID_roll_d # PID total


# SCALE PARAMETER
def MappingData(value, min, max, a, b):
    Desired_value = ((b-a)*(value-min))/(max-min)+a  #scale a value running in a range of min. max to a, b

    if Desired_value <= a:
        Desired_value = a
    elif Desired_value >= b:
        Desired_value = b

    return Desired_value

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

#############################################################################
################################## Cam 1 ####################################
# # params for controlling camera behavior
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# params for Adaptive non-maximal suppression algorithms
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.01,
                       minDistance = 250,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = [255,0,0]

# Reset and initialize all flags
HeightFlag_b = False
Pos_flag = False
ret = False
Vector = 0

def CamFaceDown():
    cap = cv2.VideoCapture(0)
    Run_Once = True
    X_avg_old = 0
    Y_avg_old = 0
    while True:
        global frame, ret, Pos_flag, HeightFlag_b, Vector, feature_params, lk_params  
        if Pos_flag == True:
            if Run_Once == True:
                ret, frame = cap.read()
                frame = rotate_image(frame, -90)
                old_frame = frame
                old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY )
                FrameW, FrameH = old_gray.shape

                # creat mask
                mask = np.zeros_like(old_gray)
                mask = cv2.rectangle(mask, (int(FrameH/12), int(FrameW/12)), (int(FrameH*11/12),int(FrameW*11/12)), (255, 255, 255), -1)
                old_gray = cv2.bitwise_and(old_gray, old_gray, mask = mask)

                # extract value and apply ANMS
                p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
                p0 = ssc(p0, 10, 0.1, FrameH, FrameW)
                p0 = np.reshape(p0, (len(p0), 1, 2))
                Run_Once = False
            
            if Run_Once == False:
                ret, frame = cap.read()
                frame = rotate_image(frame, -90)
                New_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                New_gray = cv2.bitwise_and(New_gray, New_gray, mask = mask)
                # Frame_for_drawing = New_gray.copy()

                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, New_gray, p0, None, **lk_params)

                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                    NoKPts = 0
                    New_Sum_x = 0
                    New_Sum_y = 0
                    X_avg = 0
                    Y_avg = 0
                    for i in range(len(good_new)):
                        x_projection = good_new[i][0] - int(FrameH/2)
                        y_projection = good_new[i][1] - int(FrameW/2)
                        # x_projection = good_new[i][0] - good_old[i][0]
                        # y_projection = good_new[i][1] - good_old[i][1]
                        r_projection = math.sqrt(math.pow(x_projection, 2) + math.pow(y_projection, 2))
                        diff = int(FrameW/2) - r_projection
                        if diff > 7:
                            # Frame_for_drawing = cv2.circle(Frame_for_drawing, (good_new[i][0], good_new[i][1]), 5, (255, 0, 0), -1)
                            New_Sum_x += good_new[i][0]
                            New_Sum_y += good_new[i][1]
                            NoKPts += 1
                    if NoKPts == 0:
                        Run_Once = True
                        time.sleep(0.25)
                    else:
                        X_avg = int(New_Sum_x/NoKPts)
                        Y_avg = int(New_Sum_y/NoKPts)
                        # print("x", X_avg)
                        # print("y", Y_avg)
                        Vector = [int(X_avg - int(FrameH/2)), int(Y_avg - int(FrameW/2))]
                        # print("vector", Vector)
                        # Now update the previous frame and previous points
                        X_avg_old = X_avg
                        Y_avg_old = Y_avg
                        old_gray = New_gray.copy()
                        p0 = good_new.reshape(-1, 1, 2)
                        time.sleep(0.25)

def do_nothing():
    pass

CheckListType = [1.0127, 2.054]

#############################################################################
################################ FC CONTROL #################################
def drone_control():

    # try:
    # get the curses screen window
    screen = curses.initscr()
    # term_height, term_width = screen.getmaxyx()

    # turn off input echoing
    curses.noecho()

    # respond to keys immediately (don't wait for enter)
    curses.cbreak()

    # non-blocking
    screen.timeout(0)

    # map arrow keys to special values
    screen.keypad(True)

    # screen.addstr(1, 0, "Press 'q' to quit, 'r' to reboot, 'm' to change mode, 'a' to arm, 'd' to disarm and arrow keys to control", curses.A_BOLD)
    
    # finally:
    #     # shut down cleanly
    #     curses.nocbreak(); screen.keypad(0); curses.echo()
    #     curses.endwin()
    #     if result==1:
    #         print("An error occurred... probably the serial port is not available ;)")

    # This order is the important bit: it will depend on how your flight controller is configured.
    CMDS = {
            'roll':     1500,
            'pitch':    1500,
            'yaw':      1500,
            'throttle': 1000,
            'aux1':     1000,
            'aux2':     1000
            }

    # Below it is considering the flight controller is set to use AETR.
    # The names here don't really matter, they just need to match what is used for the CMDS dictionary.
    CMDS_ORDER = ['roll', 'pitch', 'throttle', 'yaw', 'aux1', 'aux2']
    # screen.addstr(15, 0, "Connecting to the FC...")

    with MSPy(device=SERIAL_PORT, loglevel='WARNING', baudrate=115200) as board:
        # screen.addstr(15, 0, "Connecting to the FC... connected!")
        # screen.clrtoeol()
        # screen.move(1,0)

        average_cycle = deque([0]*NO_OF_CYCLES_AVERAGE_GUI_TIME)

        # It's necessary to send some messages or the RX failsafe will be activated
        # and it will not be possible to arm.
        command_list = ['MSP_API_VERSION', 'MSP_FC_VARIANT', 'MSP_FC_VERSION', 'MSP_BUILD_INFO', 
                        'MSP_BOARD_INFO', 'MSP_UID', 'MSP_ACC_TRIM', 'MSP_NAME', 'MSP_STATUS', 'MSP_STATUS_EX',
                        'MSP_BATTERY_CONFIG', 'MSP_BATTERY_STATE', 'MSP_BOXNAMES']
        if board.INAV:
            command_list.append('MSPV2_INAV_ANALOG')
            command_list.append('MSP_VOLTAGE_METER_CONFIG')

        for msg in command_list: 
            if board.send_RAW_msg(MSPy.MSPCodes[msg], data=[]):
                dataHandler = board.receive_msg()
                board.process_recv_data(dataHandler)

        # sending data in order
        slow_msgs = cycle(['MSP_STATUS_EX', 'MSP_MOTOR']) #'MSP_ANALOG', ... , ... , 'MSP_RC'

        cursor_msg = ""
        last_loop_time = last_slow_msg_time = last_cycleTime = time.time()
        while True:
            global frame, ret, Pos_flag, HeightFlag_b, Vector, CheckListType
            start_time = time.time()

            char = screen.getch() # get keypress
            curses.flushinp() # flushes buffer

            # KEY_ACTIVATION
            if char == ord('q') or char == ord('Q'):
                curses.nocbreak(); screen.keypad(0); curses.echo()
                curses.endwin()
                break

            elif char == ord('s') or char == ord('S'):
                cursor_msg = 'initialize height'
                if (len(format(board.process_mode(board.CONFIG['mode']))) == 16):
                    HeightFlag_b = True

            elif char == ord('l') or char == ord('L'):
                cursor_msg = 'position locking ...'
                if (len(format(board.process_mode(board.CONFIG['mode']))) == 16):
                    Pos_flag = True

            elif char == ord('r') or char == ord('R'):
                screen.addstr(3, 0, 'Sending Reboot command...')
                screen.clrtoeol()
                board.reboot()
                # shut down cleanly
                curses.nocbreak(); screen.keypad(0); curses.echo()
                curses.endwin()
                time.sleep(0.5)
                break

            elif char == ord('a') or char == ord('A'):
                cursor_msg = 'Sending Arm command...'
                CMDS['aux1'] = 1800

            elif char == ord('d') or char == ord('D'):
                cursor_msg = 'Sending Disarm command...'
                CMDS['aux1'] = 1000

            if char == ord('w') or char == ord('W'):
                CMDS['throttle'] = CMDS['throttle'] + 5 if CMDS['throttle'] + 5 <= 1500 else CMDS['throttle']
                cursor_msg = 'throttle(+):{}'.format(CMDS['throttle'])

            elif char == ord('e') or char == ord('E'):
                CMDS['throttle'] = CMDS['throttle'] - 5 if CMDS['throttle'] - 5 >= 1000 else CMDS['throttle']
                cursor_msg = 'throttle(-):{}'.format(CMDS['throttle'])

            # MAIN CONTROL INTERFACE
            # Adjust the height
            if HeightFlag_b == True:
                Height_Raw_Target = LiDAR_reader()
                Height_Filter_Target = Height_Raw_Target
                CMDS['throttle'] = MappingData(PID_calculation(Height_Filter_Target), -1230, 1230, 1400, 1600)
                screen.addstr(0, 0, "HEIGHT: {}".format(Height_Filter_Target), curses.A_BOLD)
                screen.clrtoeol()
                
                PID_total = PID_calculation(Height_Filter_Target)
                
                screen.addstr(1, 0, "PID_total: {}".format(PID_total), curses.A_BOLD)
                screen.clrtoeol()
                
                DesiredThrottle = round(MappingData(PID_total, -400, 400, 1500, 1650), 0)
                CMDS['throttle'] = DesiredThrottle
                cursor_msg = 'throttle_desired:{}'.format(DesiredThrottle)
                cursor_msg = 'Height(m):{0:2.2f}'.format(Height_Filter_Target)

            if Pos_flag == True:
                screen.addstr(11, 0, "Camera status : {}".format(ret), curses.A_BOLD)
                screen.addstr(12, 10, "vector : {}".format(Vector), curses.A_BOLD)
                if type(Vector) is type(CheckListType):
                    x_Error, y_Error  = Vector
                    CMDS['pitch'] = round(MappingData(PID_POS_pitch_calculation(float(y_Error * 0.0264583333)), -15, 15, 1400, 1600), 0)
                    CMDS['roll'] = round(MappingData(PID_POS_roll_calculation(float(x_Error * 0.0264583333)), -15, 15, 1400, 1600), 0)
                screen.clrtoeol()

            #############################################################
            ############### STARTING SENDING MESSESAGE ##################

            # IMPORTANT MESSAGES (CTRL_LOOP_TIME based) sending messeage
            if (time.time()-last_loop_time) >= CTRL_LOOP_TIME:
                last_loop_time = time.time()
                # Send the RC channel values to the FC
                if board.send_RAW_RC([CMDS[i] for i in CMDS_ORDER]):
                    dataHandler = board.receive_msg()
                    board.process_recv_data(dataHandler)

            # SLOW MSG processing (user GUI)
            if (time.time()-last_slow_msg_time) >= SLOW_MSGS_LOOP_TIME:
                last_slow_msg_time = time.time()

                next_msg = next(slow_msgs) # circular list

                # Read info from the FC
                if board.send_RAW_msg(MSPy.MSPCodes[next_msg], data=[]):
                    dataHandler = board.receive_msg()
                    board.process_recv_data(dataHandler)

                if next_msg == 'MSP_STATUS_EX':
                # elif next_msg == 'MSP_STATUS_EX':
                    ARMED = board.bit_check(board.CONFIG['mode'],0)
                    screen.addstr(5, 0, "ARMED: {}".format(ARMED), curses.A_BOLD)
                    screen.clrtoeol()

                    screen.addstr(6, 0, "cpuload: {}".format(board.CONFIG['cpuload']))
                    screen.clrtoeol()
                    screen.addstr(6, 50, "cycleTime: {}".format(board.CONFIG['cycleTime'])) 
                    screen.clrtoeol()

                    screen.addstr(7, 0, "mode: {}".format(board.CONFIG['mode']))
                    screen.clrtoeol()

                    screen.addstr(7, 50, "Flight Mode: {}".format(board.process_mode(board.CONFIG['mode'])))
                    screen.clrtoeol()

                elif next_msg == 'MSP_MOTOR':
                    screen.addstr(9, 0, "Motor Values: {}".format(board.MOTOR_DATA))
                    screen.clrtoeol()

                screen.addstr(11, 0, "GUI cycleTime: {0:2.2f}ms (average {1:2.2f}Hz)".format((last_cycleTime)*1000,
                                                                                            1/(sum(average_cycle)/len(average_cycle))))
                screen.clrtoeol()

                screen.addstr(3, 0, cursor_msg)
                screen.clrtoeol()

            end_time = time.time()
            last_cycleTime = end_time-start_time
            if (end_time-start_time)<CTRL_LOOP_TIME:
                time.sleep(CTRL_LOOP_TIME-(end_time-start_time))
                
            average_cycle.append(end_time-start_time)
            average_cycle.popleft()
            
if __name__ == "__main__":
    Object_1 = threading.Thread(target = drone_control)
    Object_2 = threading.Thread(target = CamFaceDown)
    Object_1.start()
    Object_2.start()
    