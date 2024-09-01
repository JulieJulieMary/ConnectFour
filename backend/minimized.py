import cv2 as cv
import numpy as np
from ultralytics import YOLO
from IPython.display import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import subprocess

#stretch image so that the border of the playable board fits in a 700x600 rectangle        
#bfs to find closest edge of the border from the start point
def Connect4BFS(matrix, start): 

    (x, y)= start
    #if start is on the border
    if matrix[x][y] > 0:
        return [y, x]

    qe = [(x, y)]
    visited = set()
    visited.add((x, y))

    height = 640
    length = 640

    #directions
    dx=[1, 0, -1, 0]
    dy=[0, 1, 0, -1]

    while qe:
        x, y = qe.pop(0)

        # Explore the neighboring nodes
        for k in range(4):
            nx = x+dx[k]
            ny = y+dy[k]
            if 0 <= nx < length and 0 <= ny < height:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    qe.append((nx, ny))
                if matrix[nx][ny] ==1:
                    return [ny, nx]                


#stretch image so that the border of the playable board fits in a 700x600 rectangle        
def Connect4GetBoard(image,matrix):

    ul= Connect4BFS(matrix, (0,0)) #left up
    ur= Connect4BFS(matrix, (0,639)) #right up
    dr= Connect4BFS(matrix, (639,639)) #right down
    dl= Connect4BFS(matrix, (639,0)) #left down 

    # image = cv.imread(image) #already given as image from parameter
    result = image.copy()

    #margin of error
    margin = -5
    # ul[0] -= margin
    ul[1] -= margin
    # ur[0] += margin
    ur[1] -= margin
    # dr[0] += margin
    dr[1] += margin
    # dl[0] -= margin
    dl[1] += margin

    #uncomment to see circled where the detected edges are
    # cv.circle(result, ul, 10, (255,0,0))
    # cv.circle(result, ur, 10, (255,0,0))
    # cv.circle(result, dr, 10, (255,0,0))
    # cv.circle(result, dl, 10, (255,0,0))

    # plt.imshow(result)
    # plt.show()
    
    #source coordinates
    src = np.array([ul, ur, dr, dl], dtype="float32")
    #destination coordinates
    dst = np.array([[0, 0], [0, 700], [600, 700],[600, 0] ], dtype="float32")
    #get transformation matrix
    m = cv.getPerspectiveTransform(src, dst)
    #wrap image using matrix
    result = cv.warpPerspective(image, m, (600, 700))
    #must rotate and flip
    result = cv.rotate(result, cv.ROTATE_90_CLOCKWISE)
    result = cv.flip(result, 1)

    return result


#makes a matrix (640x640) of points with mask contour
def Connect4MakeGrid(mask): 
  
  #coordinates of the points of first mask found
  mask = mask.xy[0]
  
  grid = np.zeros((640,640))
  for pair in mask:
    grid [int(pair[1])] [int(pair[0])] = 1

#   plt.imshow(grid)
#   plt.show()
  
  return grid


#detect piece with hsv masks
def DetectPiece(region):

    # Convert region to HSV for better color segmentation
    hsv = cv.cvtColor(region, cv.COLOR_BGR2HSV)
    
    # Define color ranges for red and yellow
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    
    # Create masks for red and yellow
    mask_red1 = cv.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv.inRange(hsv, yellow_lower, yellow_upper)
    
    # Check for the presence of each color in the region
    red_presence = np.sum(mask_red > 0)
    yellow_presence = np.sum(mask_yellow > 0)
    
    # Determine if a piece is present and its color
    if red_presence > yellow_presence and red_presence > 1200:  # threshold for minimum presence
        return 1 #red
    elif yellow_presence > red_presence and yellow_presence > 1000:
        return 2 #yellow
    else:
        return 0 #empty
    

#detect piece with mobilenetv2
def get_state_with_Ml(board,model,transform,device):

    #empty board
    game = np.zeros((6, 7)).astype(int)

    # we have a 700x600 image that we want to split into 100x100 squares, which we will then 
    # run some computer vision techniques to figure out the piece present
    for row in range (6):
        for col in range(7):
            #slice the piece region
            region = board[row*100:(row+1)*100, col*100:(col+1)*100]
            
            # Convert the region to a PIL image and apply transformations
            region_pil = Image.fromarray(cv.cvtColor(region, cv.COLOR_BGR2RGB))
            region_tensor = transform(region_pil).unsqueeze(0)  # Add batch dimension
            region_tensor = region_tensor.to(device)  # Move to GPU if available
            
            # Perform inference
            with torch.no_grad():
                outputs = model(region_tensor)
                _, predicted = torch.max(outputs, 1)  # Get the predicted class index
            
            game[row][col] = predicted.item()  # Store the prediction in the game board matrix
    
    state = cv.cvtColor(board, cv.COLOR_BGR2RGB)

    return game


#detect game state
def GetState(board):

    game = np.zeros((6, 7)).astype(int)
    # we have a 700x600 image that we want to split into 100x100 squares, which we will then 
    # run some computer vision techniques to figure out the piece present
    for row in range (6):
        for col in range(7):
            region = board[row*100:(row+1)*100, col*100:(col+1)*100]
            game[row][col] = DetectPiece(region)
    
    
    state = cv.cvtColor(board, cv.COLOR_BGR2RGB)

    return game

def scale_image(path):
        image = cv.imread(path)
        height, width, _ = image.shape
        if height == width:
            resized_image = cv.resize(image, (640, 640))
            return resized_image
        elif height > width:
            border_size = (height - width) // 2
            border = cv.copyMakeBorder(image, 0, 0, border_size, border_size, cv.BORDER_CONSTANT, value=[255, 255, 255])
            resized_border = cv.resize(image, (640, 640))
            return resized_border
        else:
            border_size = (width - height) // 2
            border = cv.copyMakeBorder(image, border_size, border_size, 0, 0, cv.BORDER_CONSTANT, value=[255, 255, 255])
            resized_border = cv.resize(image, (640, 640))
            return resized_border
        

#run code
def Connect4Run(path):
  
  image = scale_image(path)
  
  yolo_model = YOLO(model="best.pt") #load trained model, must change path
  # results = model('/content/Connect4-3/test/images') #test model and save results
  result = yolo_model(source=image,  save=False)
  if result[0].masks is None:
      print("No masks detected")
      return
  # Get masks from the results from the segmentation model
  mask = result[0].masks
  grid = Connect4MakeGrid(mask)
  board = Connect4GetBoard(image,grid)

  # Set the device to GPU 
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = models.mobilenet_v2(weights=False)
  model.classifier = nn.Sequential(
    # nn.Dropout(0.3),
    # nn.Linear(model.classifier[1].in_features, 1280), 
    # nn.ReLU(),
    # nn.Linear(1280, 128),
    # nn.Softmax(),
    # nn.Linear(128, 3)


    nn.Dropout(p=0.5),
    nn.Linear(model.classifier[1].in_features, 128),  # Increase dimensionality to match the Conv2d input
    nn.ReLU(),
    # nn.Unflatten(1, (40, 32, 1)),  # Reshape to match Conv2d input
    # nn.Conv2d(in_channels=40, out_channels=128, kernel_size=3, padding=1),
    # nn.ReLU(),
    # nn.AdaptiveAvgPool2d((1, 1)),  # Pooling to reduce dimensionality
    # nn.Flatten(),
    nn.Linear(128, 3)  # Final classification layer
  )

  model.load_state_dict(torch.load('my_model.pth'))
  model.to(device)  # Move the model to the GPU if available
  model.eval()      # Set the model to evaluation mode
  

# uncomment to save images of the pieces of the board 
#   make_squares(board)

# uncomment for old piece detection, with hsv masks
# game = GetState(board)

        # Define transforms
  transform = transforms.Compose([
      transforms.Resize((224, 224)),  # Resize images to 224x224 as required by MobileNet
      transforms.ToTensor(),          # Convert images to PyTorch tensors
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet means and stds
    ])
  
  return get_state_with_Ml(board, model,transform,device)

path = 'image.png'

game = Connect4Run(path)

for i in range(4, -1, -1):
    for j in range(7):
        if game[i][j] and not game[i + 1][j]:
            game[i][j] = 0

print("the detected board")

for line in game:
    print(*line)


playerTurn = '2'

with open('game.txt', 'w') as f:
    f.write(playerTurn)
    f.write('\n')
    for line in game:
        f.write(' '.join(map(str, line))+'\n')

move = subprocess.run(["./MCTS"], capture_output=True, text=True).stdout.strip()

for i in range (5, -1, -1):
    if not game[i][int(move)]:
        game[i][int(move)] = 3
        break

print(f"The best Move for player {playerTurn} is: {move}")

with open('move.txt', 'w') as f:
    f.write(move)
    f.write('\n')
    for line in game:
        f.write(' '.join(map(str, line))+'\n')
