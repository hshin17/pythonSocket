from socket import *
import threading
import numpy as np
import torch
import time

lock = threading.Lock()
motion_tensor = torch.Tensor()
isServerRunning = False
isGoodToGo = False
isTensorInitialized = False
Dimension = 57
Frame = 0
Scale = 1.0

def handling_recv_packet(recv_string):
    global Dimension
    global Frame
    global Scale
    tmp = list(map(float, recv_string.split()))
    lock.acquire()
    Frame = int(tmp[1])
    Scale = round(float(tmp[2]), 6)
    lock.release()
    tmp = [round(float(x),6) for x in tmp]
    
    initialize_tensor()
    for i in range(3, len(tmp), Dimension):
        single_frame_motion = []
        for j in range(Dimension):
            if i + j >= len(tmp):
                print('Debug: i is {0}, j is {1}, Dimension is {2},  Frame is {3}. But len(tmp) is {4} < Dimension * Frame + 2 {5}. len(recv_string) is {6}'.format(i,j,Dimension, Frame, len(tmp), Dimension * Frame + 2, len(recv_string)))
                print('recv_string info: {0}'.format(recv_string))
                break
            single_frame_motion.append(tmp[i + j])

        set_tensor(single_frame_motion)
    
    applying_inference()
    return

def initialize_tensor():
    lock.acquire()
    global motion_tensor, isTensorInitialized
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    motion_tensor = torch.zeros(1, 57).to(device)
    lock.release()
    isTensorInitialized = True
    return

def set_tensor(single_frame_motion):
    if len(single_frame_motion) > Dimension:
        print("Error has occured while handling motion sequence. This single frame will be omitted")
        return
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    lock.acquire()
    global motion_tensor, isTensorInitialized
    single_frame_motion = torch.unsqueeze(torch.tensor(single_frame_motion).to(device), dim=0)
    if isTensorInitialized:
        motion_tensor = single_frame_motion
        isTensorInitialized = False
    else:
        motion_tensor = torch.cat([motion_tensor, single_frame_motion], dim=0)
    lock.release()
    return

def applying_inference(): # temporary function applying inference result
    global motion_tensor
    lock.acquire()
    motion_tensor = motion_tensor * Scale
    lock.release()
    print('inference applied')
    return


def socket_test(tensor):
    return tensor * -1


def send_data(serverSocket2):
    #print('Input joint coordinates here. Input only one value to send scale value, or press Enter to terminate.')
    
    while True:
        connectionSocket, addr = serverSocket2.accept()
        
        while True:
            coord = []
            global motion_tensor
            global isGoodToGo
            
            if not isGoodToGo:
                time.sleep(0.001)
                continue
            print('in send_data function')
            lock.acquire()
            #print('lock?')
            coord = (torch.flatten(motion_tensor, 0)).tolist()
            coord = list(np.round(coord, 6))
            sending = ''
            for i in range(len(coord)):
                sending = sending + str(coord[i])
                if i < len(coord) - 1:
                    sending += ' '
            
            lock.release()
            tmp_sending = sending
            sending_packet_length = len(sending) + 1 + len(str(len(sending)))
            sending = str(sending_packet_length) + " " + sending
            if len(sending) != sending_packet_length:
                sending = str(sending_packet_length + 1) + " " + tmp_sending
            
            #print(sending)
            connectionSocket.send(sending.encode())
            connectionSocket.recv(1024)   #ack
            print('sent')
            isGoodToGo = False

def recv_all(connectionSocket):
    received = ""
    isFirstPacket = True
    totalPacketSize = 0
    
    received += connectionSocket.recv(1024).decode()
    while True:
        if isFirstPacket:
            tmp = list(map(float, received.split()))
            totalPacketSize = int(tmp[0])
            isFirstPacket = False
        else:
            if len(received) >= totalPacketSize:
                break
            received += connectionSocket.recv(1024).decode()
    
    return received


def receive_data(serverSocket):
    while True:
        global isServerRunning
        global isGoodToGo
        connectionSocket, addr = serverSocket.accept()
        received = recv_all(connectionSocket)
        while (received):
            isServerRunning = True
            #print(received)
            ack_message = "Acknowledge_from_server"
            connectionSocket.send(ack_message.encode())
            handling_recv_packet(received)
            isGoodToGo = True
            while isGoodToGo:
                time.sleep(0.001)
            received = recv_all(connectionSocket)
            
        close_msg = "Connection is closed"
        print(close_msg)
        data = []
        connectionSocket.send(close_msg.encode())
        connectionSocket.close()


def receive_func(serverSocket1, serverName='default', serverPort=50001):
    if serverName == 'default':
        serverSocket1.bind(('', serverPort))
    else:
        serverSocket1.bind((serverName, serverPort))
    serverSocket1.listen(1)
    print('The server is ready to receive. Port number is: ', serverPort)
    receive_data(serverSocket1)
    

    
def send_func(serverSocket2, serverName='default', serverPort=50002):
    if serverName == 'default':
        serverSocket2.bind(('', serverPort))
    else:
        serverSocket2.bind((serverName, serverPort))
    serverSocket2.listen(1)
    print('The server is ready to send. Port number is: ', serverPort)
    send_data(serverSocket2)

if __name__ == "__main__":
    receivingPort = 50001 #Temporary default value
    sendingPort = 50002 #Temporary default value
    serverName = '141.223.65.49' #his-smoke
    serverSocket1 = socket(AF_INET,SOCK_STREAM)
    serverSocket2 = socket(AF_INET, SOCK_STREAM)
    serverSocket1.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    serverSocket2.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

    server1 = threading.Thread(target=receive_func, args=(serverSocket1, 'default', receivingPort))  #receiver
    server2 = threading.Thread(target=send_func, args=(serverSocket2, 'default', sendingPort))  #sender

    server1.start()
    server2.start()
    print('end of main func')
