#!/usr/bin/env python3

import numpy as np
import math
import rospy
import tf

from duckietown_msgs.msg import Twist2DStamped, LanePose, Pose2DStamped, BoolStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from duckietown.dtros import DTROS, NodeType, TopicType
import EKF
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from numpy.random import uniform
from numpy.linalg import norm
from numpy.random import randn
from numpy.random import random
import scipy.stats
import random


class SensorFusionNode(DTROS):
    """
    Much of this code block is lifted from the official Duckietown Github:
    https://github.com/duckietown/dt-car-interface/blob/daffy/packages/dagu_car/src/velocity_to_pose_node.py

    The goal of this node is to provide a state estimate using one of the two filtering methods we have covered in class: the Extended Kalman Filter
    and the Particle Filter. You will be fusing the estimates from a motion model with sensor readings from the cameras.
    We have left some code here from the official Duckietown repo, but you should feel free to discard it
    if you so choose to use a different approach.

    The motion model callback as listed here will provide you with motion estimates, but you will need to figure out the covariance matrix yourself.
    Additionally, for the PF, you will need to figure out how to sample (based on the covariance matrix),
    and for the EKF, you will need to figure out how to Linearize. Our expectation is that this is a group project, so we are not providing
    skeleton code for these parts.

    Likewise, you will need to implement your own sensor model and figure out how to manage the sensor readings. We have implemented a subscriber below
    that will fire the `sensor_fusion_callback` whenever a new sensor reading comes in. You will need to figure out how to unpack the sensor reading and
    what to do with them. To do this, you might use the [tf](https://docs.ros.org/en/melodic/api/tf/html/python/tf_python.html) package to get the transformation from the tf tree
    at the appropriate time. Just like in the motion model, you will need to consider the covariance matrix for the sensor model.

    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Subscriber:
        ~velocity (:obj:`Twist2DStamped`): The robot velocity, typically obtained from forward kinematics

    Publisher:
        ~pose (:obj:`Pose2DStamped`): The integrated pose relative to the pose of the robot at node initialization

    """
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(SensorFusionNode, self).__init__(
            node_name=node_name,
            #node_type=NodeType.LOCALIZATION
            node_type=NodeType.CONTROL
        )

        # Get the vehicle name
        self.veh_name = rospy.get_namespace().strip("/")

        # Keep track of the last known pose
        self.last_pose = Pose2DStamped()
        self.last_theta_dot = 0
        self.last_v = 0

        # TODO Feel free to use a flag like this to set which type of filter you're currently using.
        # You can also define these as parameters in the launch file for this node if you're feeling fancy
        # (or if this was a system you wanted to use in the real world), but a hardcoded flag works just as well
        # for a class project like this.
        self.FUSION_TYPE = "EKF"

        # Setup the publisher
        self.path_pub = rospy.Publisher(
            "~path",
            Path,
            queue_size=1
        )
       
        self.pub_joy_override = rospy.Publisher(
            str("/" + self.veh_name + "/joy_mapper_node/joystick_override"),
            BoolStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_car_cmd = rospy.Publisher(
            str("/" + self.veh_name + "/joy_mapper_node/car_cmd"),
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_joy_cmd = rospy.Publisher(
            str("/" + self.veh_name + "/joy_mapper_node/joy"),
            Joy,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_pose2 = rospy.Publisher(
            "~pose2",
            PoseStamped,
            queue_size=1
        )
        
        self.pub_pose = rospy.Publisher(
            "~pose",
            Pose2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.LOCALIZATION
        )

        # Setup the subscriber to the motion of the robot
        self.sub_velocity = rospy.Subscriber(
            f"/{self.veh_name}/kinematics_node/velocity",
            Twist2DStamped,
            self.motion_model_callback,
            queue_size=1
        )

        # Setup the subscriber for when sensor readings come in
        self.sub_sensor = rospy.Subscriber(
            f"/{self.veh_name}/detected_tags",
            Int32MultiArray,
            self.sensor_fusion_callback,
            queue_size=1
        )
        

        self.llv = 0
        self.llt = 0
        self.llp_x = 0
        self.llp_y = 0
        self.llp_t = 0
        self.t = tf.TransformListener()
        self.kalman = None
        self.initTags = []
        self.dt = 0
        self.N = 1000
        self.x_range = [0, 1.75]
        self.y_range = [0, 1.15]
        self.hdg_range = [0, 6.28]
        self.particles = np.empty((self.N, 3))
        self.particles[:, 0] = uniform(self.x_range[0], self.x_range[1], size=self.N)
        self.particles[:, 1] = uniform(self.y_range[0], self.y_range[1], size=self.N)
        self.particles[:, 2] = uniform(self.hdg_range[0], self.hdg_range[1], size=self.N)
        self.particles[:, 2] %= 2 * np.pi
        self.std = (0.2, 0.0025)
        self.senE = 0.01
        self.weights = np.ones(self.N) / self.N
        self.mean2 = np.mean(self.particles, axis=0)
        self.trajectory = Path()
        #self.poses = []
        self.translations = []
        self.landmarks = []
        self.pose_msg = LanePose()
        self.stopping = False
        self.turnLeft = False
        self.turnRight = False
        self.leftMove = False
        self.rightMove = False
        self.finalTurn = False
        self.goals = []
        self.drawRand()
        
        self.joyCmd = Joy()
        
        self.counte = 0
	
        # ---
        self.log("Initialized.")
    
    def drawRand(self):
        x1 = random.uniform(0.55, 1.14)
        y1 = 0.15
        x2 = 1.56
        y2 = random.uniform(0.55, 1.14)
        x3 = random.uniform(0.55, 1.14)
        y3 = 1.56
        self.goals.append((x1,y1))
        self.goals.append((x2,y2))
        self.goals.append((x3,y3))
        print(self.goals)
        print("--------------------------")
    
    def sensor_fusion_callback(self, msg_sensor):
        """
        This function should handle the sensor fusion. It will fire whenever
        the robot observes an AprilTag
        """
        latest = rospy.Time(0)
        tf_exceptions = (tf.LookupException,
                        tf.ConnectivityException,
                        tf.ExtrapolationException)
        poses = []
        self.translations = []
        self.landmarks = []
        signs = []     
        try:
            for i in msg_sensor.data:
                targetFrame = "at_" + str(i) + "_camera_rgb_link"
                sourceFrame = "april_tag_cam_" + str(i)
                targetFrame1 = "april_tag_" + str(i)
                sourceFrame1 = "map"
                (translation, rot) = self.t.lookupTransform(targetFrame, sourceFrame, latest)
                (translation1, rot1) = self.t.lookupTransform(sourceFrame1, targetFrame1, latest)
                euler = euler_from_quaternion([rot[0], rot[1], rot[2], rot[3]])
                euler1 = euler_from_quaternion([rot1[0], rot1[1], rot1[2], rot1[3]])
                tmpx = translation[2]*np.cos(-euler1[2]) + translation[0]*np.sin(-euler1[2])
                tmpy = -translation[2]*np.sin(-euler1[2]) + translation[0]*np.cos(-euler1[2])
                if i == 11 or i == 32 or i == 26 or i == 61 or i == 33 or i == 31 or i == 10 or i == 24:
                    poses.append(np.array([tmpx + translation1[0], tmpy + translation1[1], euler1[2] + euler[1]]))
                self.translations.append(np.array([tmpx, tmpy, euler[1]]))
                self.landmarks.append(np.array([translation1[0], translation1[1], euler1[2]]))
                signs.append(i)
                print(i)
                print(translation)
                #print(tmpx)
                #print(tmpy)
                #print(poses)
                #print("-----------------------------------")
                              
        except tf_exceptions:
            return
        if not self.stopping:
            for i in range(0, len(signs)):
                # check if the sign we see is a stop sign
                if signs[i] == 25:
                    x_dif = self.translations[i][0]
                    y_dif = self.translations[i][1]
                    distance = math.sqrt(x_dif * x_dif + y_dif * y_dif)
                    angle_dif = self.translations[i][2] % (2 * np.pi)

                    # check if we are close enough to stop sign 
                    #if distance <= 0.2 and (angle_dif < 0.2 or angle_dif > - 0.2):
                    if distance <= 0.2:
                        # to do (try to stop then go right, then go left, then go left again) 
                        #print(distance)
                        #print(angle_dif)
                        #print("------------------------------------")
                        self.stopping = True
                        self.turnLeft = True
                    
                        # stop lane following, set override to True
                        override_msg = BoolStamped()
                        override_msg.header.stamp = rospy.Time.now()
                        override_msg.data = True
                        self.log('override_msg = False')
                        self.pub_joy_override.publish(override_msg)
                        
                        self.joyCmd.buttons[6] = 1
                        self.pub_joy_cmd.pubish(self.joyCmd)
                        
                        car_control_msg = Twist2DStamped()
                        car_control_msg.v = 0
                        car_control_msg.omega = 0
                        self.pub_car_cmd.publish(car_control_msg)
                    
        if self.stopping:
            if self.turnLeft:
                car_control_msg = Twist2DStamped()
                car_control_msg.v = 0
                car_control_msg.omega = 1.5
                self.pub_car_cmd.publish(car_control_msg)
                for i in range(0, len(signs)):
                    if signs[i] == 9:
                        angle_dif = np.abs(self.translations[i][2]) % (2 * np.pi)
                        if angle_dif < 0.1:
                            self.turnLeft = False
                            self.leftMove = True
                            car_control_msg = Twist2DStamped()
                            car_control_msg.v = 0
                            car_control_msg.omega = 0
                            self.pub_car_cmd.publish(car_control_msg)
            
            if self.leftMove:
                car_control_msg = Twist2DStamped()
                car_control_msg.v = 0.1
                car_control_msg.omega = -0.1
                self.pub_car_cmd.publish(car_control_msg)
                for i in range(0, len(signs)):
                    if signs[i] == 9:
                        x_dif = self.translations[i][0]
                        y_dif = self.translations[i][1]
                        distance = math.sqrt(x_dif * x_dif + y_dif * y_dif)
                        if distance < 0.2:
                            self.leftMove = False
                            self.turnRight = True
                            car_control_msg = Twist2DStamped()
                            car_control_msg.v = 0
                            car_control_msg.omega = 0
                            self.pub_car_cmd.publish(car_control_msg)
                           
            if self.turnRight:
                car_control_msg = Twist2DStamped()
                car_control_msg.v = 0
                car_control_msg.omega = -1.8
                self.pub_car_cmd.publish(car_control_msg)
                for i in range(0, len(signs)):
                    if signs[i] == 57:
                        angle_dif = np.abs(self.translations[i][2]) % (2 * np.pi)
                        if angle_dif < 0.1:
                            self.turnRight = False
                            self.rightMove = True
                            car_control_msg = Twist2DStamped()
                            car_control_msg.v = 0
                            car_control_msg.omega = 0
                            self.pub_car_cmd.publish(car_control_msg)
                            
            if self.rightMove:
                car_control_msg = Twist2DStamped()
                car_control_msg.v = 0.1
                car_control_msg.omega = -0.1
                self.pub_car_cmd.publish(car_control_msg)
                for i in range(0, len(signs)):
                    if signs[i] == 57:
                        x_dif = self.translations[i][0]
                        y_dif = self.translations[i][1]
                        distance = math.sqrt(x_dif * x_dif + y_dif * y_dif)
                        if distance < 0.3:
                            self.rightMove = False
                            self.finalTurn = True
                            car_control_msg = Twist2DStamped()
                            car_control_msg.v = 0
                            car_control_msg.omega = 0
                            self.pub_car_cmd.publish(car_control_msg)
                            
            if self.finalTurn:
                car_control_msg = Twist2DStamped()
                car_control_msg.v = 0
                car_control_msg.omega = 1.5
                self.pub_car_cmd.publish(car_control_msg)
                if 57 not in signs:
                    self.finalTurn = False
                    self.stopping = False
                    car_control_msg = Twist2DStamped()
                    car_control_msg.v = 0
                    car_control_msg.omega = 0
                    self.pub_car_cmd.publish(car_control_msg)
                    
                    override_msg = BoolStamped()
                    override_msg.header.stamp = rospy.Time.now()
                    override_msg.data = False
                    self.log('override_msg = False')
                    self.pub_joy_override.publish(override_msg)
            
        if self.FUSION_TYPE == "EKF":
            if self.kalman is None:
                self.initTags.extend(poses)
                if len(self.initTags) >= 10:
                    mean = np.mean(self.initTags, axis=0)
                    var = np.var(self.initTags, axis=0)
                    self.kalman = EKF.EKF(mean, var)
                    self.publishPath()
                    self.last_pose.theta = mean[2]
                    self.last_pose.x = mean[0]
                    self.last_pose.y = mean[1]
                
                return
            if len(poses) == 0:
                return 
            z = poses[0]
            for i in range(1, len(poses)):
                z = np.hstack((z, poses[i]))
            V = np.zeros((3, 2))
            if np.abs(self.llt) < 0.000001:
                V[0,0] = self.dt * np.cos(self.llp_t)
                V[2,1] = self.dt
            else:
                k1 = self.llp_t + self.dt * self.llt
                V[0,0] = (-np.sin(self.llp_t) + np.sin(k1)) / self.llt
                V[0,1] = self.llv * (np.sin(self.llp_t) - np.sin(k1)) / (self.llt * self.llt) + self.llv * np.cos(k1)* self.dt /self.llt
                V[1,0] = (np.cos(self.llp_t) - np.cos(k1)) / self.llt
                V[2,0] = -self.llv * (np.cos(self.llp_t) - np.cos(k1)) / (self.llt * self.llt) + self.llv * np.sin(k1)* self.dt /self.llt
                V[2,1] = self.dt
            M = np.zeros((2,2))
            M[0,0] = (self.llt * self.llt) + (self.llv * self.llv)
            M[1,1] = (self.llt * self.llt) + (self.llv * self.llv)
            R = np.matmul(np.matmul(V, M), np.transpose(V))
            R = R * 0.1
            Q = np.eye(3) * 0.01
            #tmp = np.var(poses, axis=0)
            #Q[0,0] = tmp[0]
            #Q[1,1] = tmp[1]
            #Q[2,2] = tmp[2]
            predicted = np.array([self.last_pose.x, self.last_pose.y, self.last_pose.theta])
            if self.dt != 0 and (self.llv != 0 or self.llt != 0): 
                self.kalman.update(z, Q, R, self.dt, predicted, self.llv, self.llt)
                self.last_pose.x = self.kalman.mean[0]
                self.last_pose.y = self.kalman.mean[1]
                self.last_pose.theta = self.kalman.mean[2]
                if len(self.goals) > 0:
                    for curX, curY in self.goals:
                        curDistance = (self.kalman.mean[0] - curX) * (self.kalman.mean[0] - curX) + (self.kalman.mean[1] - curY) * (self.kalman.mean[1] - curY)
                        curDistance = math.sqrt(curDistance)
                        if curDistance < 0.3:
                            override_msg = BoolStamped()
                            override_msg.header.stamp = rospy.Time.now()
                            override_msg.data = True
                            self.log('override_msg = False')
                            self.pub_joy_override.publish(override_msg)
                        
                            self.joyCmd.buttons[6] = 1
                            self.pub_joy_cmd.pubish(self.joyCmd)
                            car_control_msg = Twist2DStamped()
                            car_control_msg.v = 0
                            car_control_msg.omega = 0
                            self.pub_car_cmd.publish(car_control_msg)
                            self.counte = self.counte + 1
                            print(curX)
                            print(curY)
                            print("--------------------")
                            if self.counte > 20:
                                self.goals.remove((curX, curY))
                                self.counte = 0
                                override_msg = BoolStamped()
                                override_msg.header.stamp = rospy.Time.now()
                                override_msg.data = False
                                self.log('override_msg = False')
                                self.pub_joy_override.publish(override_msg)
                            
                self.publishPath()
        else:
            return
                
    def weightedMean(self):
        mean = np.average(self.particles, weights=self.weights, axis=0)
        return mean
        
    def publishPath(self):
        tmp1 = quaternion_from_euler(0.0, 0.0, self.kalman.mean[2]) 
        poseS = PoseStamped()
        poseS.pose.position.x = self.kalman.mean[0]
        poseS.pose.position.y = self.kalman.mean[1]
        poseS.pose.position.z = 0.0
        poseS.pose.orientation.x = tmp1[0]
        poseS.pose.orientation.y = tmp1[1]
        poseS.pose.orientation.z = tmp1[2]
        poseS.pose.orientation.w = tmp1[3]
        poseS.header.frame_id = "map"
        poseS.header.stamp = rospy.Time.now()
        self.pub_pose2.publish(poseS)
        
        self.trajectory.header.frame_id = "map"
        self.trajectory.header.stamp = rospy.Time.now()
        
        self.trajectory.poses.append(poseS)
        self.path_pub.publish(self.trajectory)
        
    def publishPath2(self):
        tmp1 = quaternion_from_euler(0.0, 0.0, self.mean2[2]) 
        poseS = PoseStamped()
        poseS.pose.position.x = self.mean2[0]
        poseS.pose.position.y = self.mean2[1]
        poseS.pose.position.z = 0.0
        poseS.pose.orientation.x = tmp1[0]
        poseS.pose.orientation.y = tmp1[1]
        poseS.pose.orientation.z = tmp1[2]
        poseS.pose.orientation.w = tmp1[3]
        poseS.header.frame_id = "map"
        poseS.header.stamp = rospy.Time.now()
        self.pub_pose2.publish(poseS)
        
        self.trajectory.header.frame_id = "map"
        self.trajectory.header.stamp = rospy.Time.now()
        
        self.trajectory.poses.append(poseS)
        self.path_pub.publish(self.trajectory)
         
    def motion_model_callback(self, msg_velocity):
        """

        This function will use robot velocity information to give a new state
        Performs the calclulation from velocity to pose and publishes a messsage with the result.


        Feel free to modify this however you wish. It's left more-or-less as-is
        from the official duckietown repo

        Args:
            msg_velocity (:obj:`Twist2DStamped`): the current velocity message

        """
        if self.last_pose.header.stamp.to_sec() > 0:  # skip first frame

            self.dt = (msg_velocity.header.stamp - self.last_pose.header.stamp).to_sec()
            land = np.copy(self.landmarks)
            trans = np.copy(self.translations)
            

            # Integrate the relative movement between the last pose and the current
            theta_delta = self.last_theta_dot * self.dt
            # to ensure no division by zero for radius calculation:
            if np.abs(self.last_theta_dot) < 0.000001:
                # straight line
                x_delta = self.last_v * self.dt
                y_delta = 0
            else:
                # arc of circle
                radius = self.last_v / self.last_theta_dot
                x_delta = radius * np.sin(theta_delta)
                y_delta = radius * (1.0 - np.cos(theta_delta))

            # Add to the previous to get absolute pose relative to the starting position
            theta_res = self.last_pose.theta + theta_delta
            x_res = self.last_pose.x + x_delta * np.cos(self.last_pose.theta) - y_delta * np.sin(self.last_pose.theta)
            y_res = self.last_pose.y + y_delta * np.cos(self.last_pose.theta) + x_delta * np.sin(self.last_pose.theta)
            
            
            if self.FUSION_TYPE == "PF":
                if self.dt != 0 and len(trans) != 0 and len(land) != 0 and (np.abs(self.last_theta_dot) > 0.000001 or np.abs(self.last_v) > 0.000001):
                    robotx = []
                    roboty = []
                    robotz = []
                    for i in trans:
                        robotx.append(i[0])
                        roboty.append(i[1])
                        robotz.append(i[2])
                    robotx = robotx + randn(len(land)) * self.senE
                    roboty = roboty + randn(len(land)) * self.senE
                    robotz = (robotz + randn(len(land)) * self.std[0]) % (2*np.pi)
                    i = 0
                    #print(land)
                    #print(trans)
                    self.particles[:, 0] += x_delta * np.cos(self.particles[:, 2]) - y_delta * np.sin(self.particles[:, 2]) + (randn(self.N) * self.std[1]) * np.cos(self.particles[:, 2])
                    self.particles[:, 1] += y_delta * np.cos(self.particles[:, 2]) + x_delta * np.sin(self.particles[:, 2]) + (randn(self.N) * self.std[1]) * np.sin(self.particles[:, 2])
                    self.particles[:, 2] += theta_delta + (randn(self.N) * self.std[0])
                    self.particles[:, 2] %= 2 * np.pi
                    for landm in land:
                        xs = []
                        ys = []
                        zs = []
                        for p in range(self.N):
                            xs.append(self.particles[p][0] - landm[0])
                            ys.append(self.particles[p][1] - landm[1])
                            zs.append((self.particles[p][2] - landm[2]) % (2*np.pi))
                            #if p == 0:
                                #print(xs)
                                #print(ys)
                                #print(zs)
                                #print(self.particles[p])
                        
                        self.weights *= scipy.stats.norm(xs, self.senE).pdf(robotx[i])
                        self.weights *= scipy.stats.norm(ys, self.senE).pdf(roboty[i])
                        self.weights *= scipy.stats.norm(zs, self.senE).pdf(robotz[i])
                        i = i + 1
                    
                    self.weights += 1.e-300
                    self.weights /= sum(self.weights)
        
                    eff = 1. / np.sum(np.square(self.weights))
                    if eff < self.N/2:
                        cumulative_sum = np.cumsum(self.weights)
                        cumulative_sum[-1] = 1.
                        indexes = np.searchsorted(cumulative_sum, random(self.N))

                        self.particles[:] = self.particles[indexes]
                        self.weights = self.weights[indexes]
                        #self.weights.fill(1.0 / self.N)
                        self.weights /= sum(self.weights)
    
                    self.mean2 = self.weightedMean()
                    #print(self.mean2)
                    #print("---------------------------------------------")
                    self.publishPath2()

            # Update the stored last pose
            self.llp_x = self.last_pose.x
            self.llp_y = self.last_pose.y
            self.llp_t = self.last_pose.theta
            self.last_pose.theta = theta_res
            self.last_pose.x = x_res
            self.last_pose.y = y_res
            

            # TODO Note how this puts the motion model estimate into a message and publishes the pose.
            # You will also need to publish the pose coming from sensor fusion when you correct
            # the estimates from the motion model
            msg_pose = Pose2DStamped()
            msg_pose.header = msg_velocity.header
            msg_pose.header.frame_id = self.veh_name
            msg_pose.theta = theta_res
            msg_pose.x = x_res
            msg_pose.y = y_res
            self.pub_pose.publish(msg_pose)
            
        self.llt = self.last_theta_dot
        self.llv = self.last_v
        self.last_pose.header.stamp = msg_velocity.header.stamp
        self.last_theta_dot = msg_velocity.omega
        self.last_v = msg_velocity.v



if __name__ == '__main__':
    # Initialize the node
    sensor_fusion_node = SensorFusionNode(node_name='sensor_fusion_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
