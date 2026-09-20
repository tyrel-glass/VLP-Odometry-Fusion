import matplotlib.pyplot as plt
import socket
import time
import serial
import time
# import paper2_navigation.data.triad_openvr.triad_openvr as vr
import pandas as pd
import numpy as np
import csv
# import openvr
import os
from scipy.spatial.transform import Rotation as R



#====================================================================
#                   ----- VLP FUNCTIONS ------
#====================================================================

def average_vlp_readings(data):

    return average_of_closest_to_median(data, num_points= 5)

def process_vlp(df):
    df.loc[:,'vlp_data'] = df['vlp_data'].apply(lambda v: np.array([re for re in eval(v) if re is not None]))

    df.loc[:,'pks'] = df['vlp_data'].apply(lambda v: [np.array(calc_pks(FFT_win(data)[0]))for data in v])

    df.loc[:,['L1', 'L2', 'L3', 'L4']] = np.array(df['pks'].apply(lambda v: np.apply_along_axis(np.mean, 0,np.array(v))).to_list())

    # ======================================================================
    # ==============================  TESTING   =======================
    # ======================================================================
    # ======================================================================
    df.loc[:,['L1_test', 'L2_test', 'L3_test', 'L4_test']] = np.array(df['pks'].apply(lambda v: np.apply_along_axis(np.median, 0,np.array(v[:1]))).to_list())
    # ======================================================================
    # ======================================================================


    return df

def FFT_win(clip):
    signal = np.array(clip)
    signal = signal - np.mean(signal)
    signal = signal/np.max(signal)
    win = np.hanning(len(signal))
    signal = signal*win

    fft =  np.fft.fft(signal)
    fft = fft[:int(len(fft)/2)]
    fft = np.abs(fft)
    fre = np.linspace(0,25200,int(len(fft)))

    return fft, fre


def take_mean_fft(n):
    ffts = []
    print('taking vlp readings: ',end = ' ')
    for i in range(n):
        print(i, end = ' ')
        fft, fre = FFT_win(read_vlp())
        ffts.append(fft)
        time.sleep(0.1)

    return np.mean(ffts, axis = 0), fre

def read_n_vlp(n):
    adcs = []
    print('taking vlp readings: ',end = ' ')
    for i in range(n):
        print(i, end = ' ')
        adc = read_vlp()
        adcs.append(adc)
        time.sleep(0.1) #give esp a rest
    return adcs

def calc_pks(fft, width = 8):
    light_frequencys  = {
    'l1':15000,
    'l2':17000,
    'l3':20000,
    'l4':23000,
    }
    fft = np.array(fft)

    fre = np.linspace(0,25000,int(len(fft)))
    intensitys = []

    for light in light_frequencys.keys():
        cen_fre = light_frequencys[light]
        cen_ind = len(fre[fre<cen_fre])
        lower = cen_ind - width
        upper = cen_ind + width
        index = fft[lower:upper].argmax() + lower
        value = fft[index]
        intensitys.append(value)
    return intensitys


def read_vlp(max_retries=3, timeout=3):
    ESP32_IP="192.168.10.101"  #old board
    # ESP32_IP="192.168.10.104"  #new board
    ESP32_PORT = 8080
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Attempt to connect to the ESP32
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)  # Set timeout for the connection
            s.connect((ESP32_IP, ESP32_PORT))
            s.sendall(b"FETCH\n")  # Request ADC data

            adc_data = []  # List to store parsed ADC values
            buffer = ""  # Buffer to accumulate partial data
            start_received = False  # Flag to check if START marker is processed

            while True:
                chunk = s.recv(1024).decode().strip()  # Receive and decode a chunk
                buffer += chunk  # Accumulate the chunk into the buffer

                # Process the buffer
                if "START ADC DATA" in buffer:
                    start_received = True
                    buffer = buffer.split("START ADC DATA", 1)[1]  # Discard the marker

                if "END ADC DATA" in buffer:
                    # Process up to the "END ADC DATA" marker
                    data_section, _, buffer = buffer.partition("END ADC DATA")

                    if start_received:
                        data_section = data_section.replace('\r\n', '')  # Remove newlines
                        try:
                            # Split the string into integers, skipping empty or invalid values
                            adc_data = [int(val) for val in data_section.split(',') if val.strip().isdigit()]
                        except ValueError as e:
                            print(f"Error parsing adc data: {data_section} - {e}")  # Debugging invalid lines

                    break  # Exit the loop once the end marker is processed

            s.close()  # Close the socket connection
            return adc_data

        except (socket.timeout, ConnectionError) as e:
            retry_count += 1
            print(f"Connection attempt {retry_count} failed: {e}")
            time.sleep(1)  # Wait 1 second before retrying

    print("Failed to connect to ESP32 after multiple attempts.")
    return None  # Return None if all retries fail



def uart_vlp_read():
    com_port = '/dev/tty.usbserial-210'
    vlp_port = serial.Serial(com_port, 460800, timeout= 20)
    vlp_port.write(b'b\n')
    adc_data = []  # List to store parsed ADC values
    start_received = False  # Flag to check if START marker is processed
    while True:
        line = vlp_port.readline().decode().strip()  # Receive and decode a chunk
        # Process the buffer
        if "ADC Data:[" in line:
            vlp_port.write(b's\n')
            break  # Exit the loop once the end marker is processed

    line = line.split('ADC Data:[',1)[1]
    line = line.split(']')[0]
    adc_data = [int(val) for val in line.split(',') if val.strip().isdigit()]
    vlp_port.close()
    return adc_data




#====================================================================
#                   ----- VIVE FUNCTIONS ------
#====================================================================
def vive_setup():
    # Initialize OpenVR
    # ovr = openvr.init(openvr.VRApplication_Scene)
    v = vr.triad_openvr()
    print(v.devices)
    return v

def read_vive(vive, n_readings = 10 ):
    readings = []
    for _ in range(n_readings):
        readings.append(np.array(vive.devices["tracker_1"].get_pose_matrix()))
        time.sleep(0.1)
    # mean_readings = np.mean(readings, axis = 0)
    return readings


def take_vive_cal_point(point_no, log_file, vive, raw = True):
    vive_data = read_vive(vive)
    cmd = 'CAL:'+str(point_no)
    vive_robot_log_write(vive_data =vive_data,vlp_data = None, cmd= cmd, log_file= log_file)

def get_last_vive_position(log_file):
    # Load the log file
    df = pd.read_csv(log_file, delimiter='|', header=0)
    # Extract the vive_data from the last row
    df = df.iloc[-1:]
    vive_data = parse_vive(df)
    # mean_readings = np.array(eval(last_row_vive_data.replace('array', 'np.array'))).mean(axis = 0)
    return vive_data['vive_data'].iloc[0]

def parse_vive(df):

        def average_vive_matrix(list_of_matricies):
                list_of_matricies = [a['m'] for a in list_of_matricies]  #convert to normal array
                return np.mean(list_of_matricies, axis = 0)

        # convert from string
        df['vive_data'] = df['vive_data'].apply( lambda v: np.array(eval(v.replace('array','np.array'))))

        def check_for_none_array(v):
                return [m for m in v if m.tolist() is not None]

        df['vive_data'] = df['vive_data'].apply(check_for_none_array)

        row_filt = df['vive_data'].notna()
        df.loc[row_filt, 'vive_data'] = df['vive_data'][row_filt].apply(average_vive_matrix)

        return df

def transform_vive_df(df):
    transformer = ViveToRobotTransform()
    transformer.derive_transform(df)

    na_filter = df['vive_data'].notna()

    df.loc[na_filter,'transformed_vive'] = df['vive_data'].apply(lambda d: transformer.transform_pose(add_bottom_row(d)))

    vive_labels = ['vive_x', 'vive_y', 'vive_z', 'vive_yaw', 'vive_pitch','vive_roll']
    df.loc[na_filter, vive_labels] =  np.array(df.loc[na_filter,'transformed_vive'].apply(extract_pose_y_up).to_list())

    return df


######################## VIVE TRANSLANTION CODE #########################
class ViveToRobotTransform:
    def __init__(self):
        self.transformation_matrix = None

    def derive_transform(self, df):
        """
        Derive the transformation matrix using pose matrices (4x4).
        """
        # Extract calibration points
        calibration_points = df[df['last_cmd'].str.startswith('CAL')].head(3)
        assert len(calibration_points) == 3, "Insufficient calibration points for alignment."

        # Known robot pose matrices for calibration points (identity rotation for simplicity in this example)
        robot_poses = [
            np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0.9],
                      [0, 0, 0, 1]]),  # CAL:1
            np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]),  # CAL:2
            np.array([[1, 0, 0, 0.9],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])   # CAL:3
        ]

        # Vive pose matrices (ensure homogeneous 4x4 matrices)
        vive_poses = calibration_points['vive_data'].apply(np.array).to_list()

        # Compute the transformation matrix using point-to-point alignment
        robot_matrix = np.array(robot_poses)
        vive_matrix = np.array(vive_poses)

        # Solve for the transformation matrix
        self.transformation_matrix = self.align_poses(vive_matrix, robot_matrix)

    def align_poses(self, vive_poses, robot_poses):
        """
        Compute the best-fit transformation matrix between two sets of pose matrices.
        """
        # Compute centroids
        vive_centroid = np.mean([pose[:3, 3] for pose in vive_poses], axis=0)
        robot_centroid = np.mean([pose[:3, 3] for pose in robot_poses], axis=0)

        # Translate poses to origin (center them)
        vive_centered = [pose.copy() for pose in vive_poses]
        robot_centered = [pose.copy() for pose in robot_poses]

        for vc, rc in zip(vive_centered, robot_centered):
            vc[:3, 3] -= vive_centroid
            rc[:3, 3] -= robot_centroid

        # Compute rotation using Kabsch algorithm
        vive_stack = np.hstack([vc[:3, 3].reshape(-1, 1) for vc in vive_centered])
        robot_stack = np.hstack([rc[:3, 3].reshape(-1, 1) for rc in robot_centered])
        H = vive_stack @ robot_stack.T
        U, _, Vt = np.linalg.svd(H)
        rotation_matrix = U @ Vt

        # Check for reflection
        if np.linalg.det(rotation_matrix) < 0:
            Vt[-1, :] *= -1
            rotation_matrix = U @ Vt

        # Translation is the difference between centroids
        translation_vector = robot_centroid - rotation_matrix @ vive_centroid

        # Combine into a single 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector

        return transformation_matrix

    def transform_pose(self, pose):
        """
        Transform a 4x4 pose matrix using the derived transformation.
        """
        if self.transformation_matrix is None:
            raise ValueError("Transformation not yet derived. Call `derive_transform` first.")

        return self.transformation_matrix @ pose


def extract_euler_angles(rotation_matrix, sequence='xyz', degrees=False):

    # Check if the matrix is 3x3
    if rotation_matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")

    # Convert the rotation matrix into a Rotation object
    r = R.from_matrix(rotation_matrix)

    # Extract Euler angles using the specified sequence
    euler_angles = r.as_euler(sequence, degrees=degrees)

    return euler_angles


def extract_pose_y_up(pose_mat):

    import numpy as np
    import math
    # Translation components (x, y, z)
    x = pose_mat[0, 3]
    y = pose_mat[1, 3]
    z = pose_mat[2, 3]

    # Rotation matrix components
    R = pose_mat[:3, :3]
    yaw, pitch, roll = extract_euler_angles(R, sequence='zyx', degrees=True)
    return [x, y, z, yaw, pitch, roll]

def add_bottom_row(matrix_3x4):
    """
    Add the bottom row [0, 0, 0, 1] to a 3x4 pose matrix to create a 4x4 pose matrix.
    """
    bottom_row = np.array([[0, 0, 0, 1]], dtype=matrix_3x4.dtype)
    matrix_4x4 = np.vstack((matrix_3x4, bottom_row))
    return matrix_4x4


#====================================================================
#                   ----- CNC FUNCTIONS ------
#====================================================================

def relative_movement(x, y, cnc_serial):
    """Perform relative movement on X and Y axes."""
    try:
        # Switch to relative positioning mode
        cnc_serial.write(b"G91\n")  # G91: Relative positioning
        cnc_serial.readline()  # Wait for CNC response

        # Send the movement command directly
        gcode_command = f"G1 X{x} Y{y} F1000"
        print(f"Generated G-code: {gcode_command}")
        cnc_serial.write(str.encode(gcode_command) + b"\n")
        grbl_out = cnc_serial.readline().decode().strip()
        print("GRBL Response: ", grbl_out)

        # Return to absolute positioning mode
        cnc_serial.write(b"G90\n")  # G90: Absolute positioning
        cnc_serial.readline()  # Wait for CNC response
    except Exception as e:
        print(f"Error during relative movement: {e}")

def absolute_movement(x, y, cnc_serial, feedrate=1000):
    """
    Moves the CNC to an absolute location (X, Y) at the specified feedrate.

    Args:
        x (float): Absolute X-coordinate.
        y (float): Absolute Y-coordinate.
        feedrate (int): Feedrate for the movement in units per minute. Default is 1000.

    Returns:
        None
    """
    try:
        # Ensure absolute positioning mode
        cnc_serial.write(b"G90\n")  # G90: Absolute positioning mode
        cnc_serial.readline()  # Wait for CNC response

        # Create and send the absolute movement command
        gcode_command = f"G1 X{x} Y{y} F{feedrate}"
        print(f"Sending command: {gcode_command}")
        cnc_serial.write(str.encode(gcode_command) + b"\n")

        # Wait for CNC's acknowledgment
        grbl_out = cnc_serial.readline().decode().strip()
        print("GRBL Response: ", grbl_out)
    except Exception as e:
        print(f"Error during absolute movement: {e}")


def init_cnc(cnc_serial):
    # Wake up GRBL and zero CNC
    cnc_serial.write(b"\r\n\r\n")
    time.sleep(2)
    cnc_serial.flushInput()
    cnc_serial.write(str.encode("$X") + b"\n")
    cnc_serial.readline()  # Wait for CNC response
    cnc_serial.write(str.encode("G10 P0 L20 X0 Y0 Z0") + b"\n")
    cnc_serial.readline()  # Zero CNC response


#====================================================================
#                   ----- LOGGING FUNCTIONS ------
#====================================================================


# ----     CNC    -----
def cnc_log_clear(file_name):
    """Clears the log file and writes headers with a pipe delimiter."""
    with open(file_name, 'w') as f:
        f.write("position_id|timestamp|vive_data|vlp_data|cnc_data\n")

def cnc_log_write(file_name, reading_no, x, y, vive_data, vlp_data):
    """Writes a single data point to the log file."""
    with open(file_name, 'a') as f:
        f.write(f'{reading_no}|')
        f.write(f'{time.time()}|')
        f.write(f'"{vive_data}"|')  # Enclose in quotes to handle special characters
        f.write(f'"{vlp_data}"|')  # Enclose in quotes to handle special characters
        f.write(f'{x},{y}\n')

def get_last_logged_point(log_file):
    """Returns the last recorded position_id if log file exists, otherwise returns -1."""
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        data = pd.read_csv(log_file, sep='|')
        if not data.empty:
            return data['position_id'].iloc[-1]  # Get the last position_id
    return -1


# ----     ROBOT    -----
def vive_robot_log_clear(log_file = 'robot_vive_data_log.csv' ):
    """Clears the log file and writes headers with a pipe delimiter."""
    with open(log_file, 'w') as f:
        f.write("vive_data|vlp_data|last_cmd\n")


def vive_robot_log_write(vive_data,vlp_data, cmd, log_file = 'robot_vive_data_log.csv'):
    """Writes a single data point to the log file."""
    vive_data = str(vive_data).replace('\n', '')  # Remove newline characters
    last_cmd = str(cmd).replace('\n', '')    # Remove newline characters

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([vive_data,vlp_data, cmd])

#====================================================================
#                   ----- NAVIGATION FUNCTIONS ------
#====================================================================

def generate_scan_points(step=50, width=900, height=1000):
    """
    Generates a set of coordinate points to scan over a 2D area with a zigzag pattern.
    """
    points = []
    for y in range(0, height + step, step):  # Increment along the height
        if y % (2 * step) == 0:
            # Move left to right
            row_points = [(x, y) for x in range(0, width + step, step)]
        else:
            # Move right to left
            row_points = [(x, y) for x in range(width, -step, -step)]
        points.extend(row_points)
    return np.array(points)


#====================================================================
#                   ----- ROBOT FUNCTIONS ------
#====================================================================



def send_command_to_nano(command):
    """
    Sends a command to the ESP server, which forwards it to the Arduino Nano,
    and retrieves the response.

    :param command: Command string to send to the Nano.
    :return: Response string from the Nano.
    """
    # ESP Server configuration
    ESP_IP = "192.168.10.102"  # Replace with the ESP's IP address
    ESP_PORT = 8080           # Port number defined in the ESP server code
    try:
        # Create a TCP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            # Set a timeout of 30 seconds
            client_socket.settimeout(30.0)

            # Connect to the ESP server
            client_socket.connect((ESP_IP, ESP_PORT))
            print(f"Connected to ESP server at {ESP_IP}:{ESP_PORT}")

            # Send the command
            client_socket.sendall((command + '\n').encode('utf-8'))

            # Receive the response
            response = client_socket.recv(1024).decode('utf-8')
            print(f"Response received: {response}")
            return response
    except socket.timeout:
        print("Error: Operation timed out.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None




def process_move_wifi(cmd, log_file, vive, transformer = None, vlp_read = True, log = True):

    nano_response = send_command_to_nano(cmd)
    print(nano_response)
    nano_response = send_command_to_nano('')

    if vlp_read:

        vlp_data = read_n_vlp(3)
    else:
        vlp_data = None
    if log:
        vive_data = read_vive(vive)
        vive_robot_log_write(vive_data,vlp_data, cmd = cmd, log_file = log_file)


#====================================================================
#                   ----- HELPER FUNCTIONS ------
#====================================================================



def average_of_closest_to_median(data, num_points=5):
    """
    Calculate the average of the closest `num_points` to the median in an array.

    Parameters:
        data (list or array-like): The input array of numbers.
        num_points (int): The number of points closest to the median to consider.

    Returns:
        float: The average of the closest `num_points` to the median.
    """
    if len(data) < num_points:
        raise ValueError("The number of points to average must be less than or equal to the size of the array.")

    # Step 1: Calculate the median
    median = np.median(data)

    # Step 2: Calculate absolute differences from the median
    differences = [(x, abs(x - median)) for x in data]

    # Step 3: Sort the array by the differences
    sorted_by_difference = sorted(differences, key=lambda x: x[1])

    # Step 4: Select the `num_points` closest to the median
    closest_points = [x[0] for x in sorted_by_difference[:num_points]]

    # Step 5: Calculate the average of the selected points
    return np.mean(closest_points)



def process_cnc(df):
    df[['cnc_y', 'cnc_x']] = df['cnc_data'].str.split(',', expand=True).astype(float)
    return df


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def check_for_none_array(v):
    if len(v[0].reshape(-1)) == 1:
        return None
    else:
        return v

    # ----- ROBOT SERIAL FUNCTIONS -----
def open_robot_port(com_port = 'COM10'):
    robot_port = serial.Serial(com_port, 9600, timeout= 20)
    print(robot_port.readline())
    return robot_port

def send_robot_cmd(robot_ser, cmd):
    robot_ser.write((cmd + '\n').encode())
    print(robot_ser.readline())
    print(robot_ser.readline())



########################  CONTROL PROCESSING ##########################

def process_move(cmd, robot_ser, log_file, vive, transformer):

    send_robot_cmd(robot_ser = robot_ser, cmd = cmd)
    vive_data = read_vive(vive, transformer= transformer)
    vive_robot_log_write(vive_data, cmd = cmd, log_file = log_file)



def normalize_angle(angle):
    """
    Normalizes an angle to the range [-180, 180] degrees.

    :param angle: Angle in degrees (float or int).
    :return: Normalized angle in degrees.
    """

    def compute(a):
        # Bring the angle within the range [0, 360]
        # Adjust to [-180, 180] range
        a = a % 360
        if a > 180:
            a -= 360
        return a

    a_type = type(angle)
    if (a_type == np.ndarray) or (a_type == list):
        new_lst = []
        for i in range(len(angle)):
            new_lst.append(compute(angle[i]))
        return new_lst
    else:
        return compute(angle)
