from flask import Flask, request, jsonify
import os, glob
from sklearn import preprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

app = Flask(__name__)

UPLOAD_FOLDER = 'orl_faces/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}
namefile = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'

def cropt_img(dir, filename, type):
    if type == 0:
        folder = 'orl_faces/' + dir + filename
    else:
        folder = 'absent/' + filename
    imgcropt  = cv2.imread(folder)
    graycropt  = cv2.cvtColor(imgcropt, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(namefile)
    facescropt = face_cascade.detectMultiScale(graycropt , 1.1, 4)

    for (x, y, w, h) in facescropt:
        cv2.rectangle(imgcropt, (x, y), (x+w, y+h), 
                    (0, 0, 255), 2)
        
        facescropt = imgcropt[y:y + h, x:x + w]
        cv2.imwrite(folder, facescropt)
    return 'sucssess'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#load model for prediction
# model = pickle.load(open('model.pkl','rb'))

def file_save(array,errors,success,folderReq = None):
    for file in array:
        if file and allowed_file(file.filename):
            folder = 'orl_faces/'
            i = 0
            if folderReq is None:
                for i in range(1000000):
                    subFile = 's{}/'.format(i)
                    fullFolder = folder+subFile
                    isFile = os.path.isdir(fullFolder)
                    if not isFile:
                        makeDir = os.path.join(folder, subFile)
                        os.mkdir(makeDir)
                        file.save(os.path.join(fullFolder, file.filename))
                        success = True
                        errors = {}
                        return subFile, success, errors, file.filename
            else:
                fullFolder = folder+folderReq
                isFile = os.path.isdir(fullFolder)
                print(isFile)
                if isFile:
                    file.save(os.path.join(fullFolder, file.filename))
                    success = True
                    errors = {}
                    return folderReq, success, errors, file.filename
        else:
            errors["message"] = 'File type of {} is not allowed'.format(file.filename)
            success = False
            return file.filename, success, errors, file.filename

def success_check(success, errors):
    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp
        
@app.route('/')
def main():
    return 'This is API'

@app.route('/api/save_image', methods=['POST'])
def img_save():
    def file_save(array,errors,success,folderReq = None):
        for file in array:
            if file and allowed_file(file.filename):
                folder = 'orl_faces/'
                i = 0
                if folderReq is None:
                    for i in range(1000000):
                        subFile = 's{}/'.format(i)
                        fullFolder = folder+subFile
                        isFile = os.path.isdir(fullFolder)
                        if not isFile:
                            makeDir = os.path.join(folder, subFile)
                            os.mkdir(makeDir)
                            file.save(os.path.join(fullFolder, file.filename))
                            success = True
                            errors = {}
                            return subFile, success, errors, file.filename
                else:
                    fullFolder = folder+folderReq
                    isFile = os.path.isdir(fullFolder)
                    print(isFile)
                    if isFile:
                        file.save(os.path.join(fullFolder, file.filename))
                        success = True
                        errors = {}
                        return folderReq, success, errors, file.filename
            else:
                errors["message"] = 'File type of {} is not allowed'.format(file.filename)
                success = False
                return file.filename, success, errors, file.filename
    
        
    value = 0
    errors0 = {}
    success0 = False
    direktori0 = ''
    if 'file'.format(value) not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    direktori0 = request.files.getlist('file')
    folder = request.form.get('folder')
    folderReq = folder
    direktori0, success0, errors0, filename0 = file_save(direktori0,errors0,success0,folder)

    print(direktori0)
    namefile = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    folder0 = 'orl_faces/' + direktori0 + filename0
    imgcropt  = cv2.imread(folder0)
    graycropt  = cv2.cvtColor(imgcropt, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(namefile)
    facescropt = face_cascade.detectMultiScale(graycropt , 1.1, 4)
    for (x, y, w, h) in facescropt:
        cv2.rectangle(imgcropt, (x, y), (x+w, y+h), 
                    (0, 0, 255), 2)
        
        facescropt = imgcropt[y:y + h, x:x + w]
        cv2.imwrite(folder0, facescropt)
            
    if not success0:
        resp = jsonify(errors0)
        resp.status_code = 400
        return resp
    
    return jsonify(isError=False, message="Success", statusCode=200, pythonDirektory = direktori0), 200

@app.route('/api/eigenface', methods=['POST'])
def recognize_image():
    if 'absent' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('absent')
    filename = ''
    for file in files:
        file.save(os.path.join('absent/', file.filename))
        filename = 'absent/' + file.filename
        imgcropt  = cv2.imread(filename)
        graycropt  = cv2.cvtColor(imgcropt, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(namefile)
        facescropt = face_cascade.detectMultiScale(graycropt , 1.1, 4)
        for (x, y, w, h) in facescropt:
            cv2.rectangle(imgcropt, (x, y), (x+w, y+h), 
                        (0, 0, 255), 2)
            facescropt = imgcropt[y:y + h, x:x + w]
            cv2.imwrite(filename, facescropt)
    
    dataset_path = os.getcwd() + '/orl_faces/' #jalur direktori ke kumpulan data yang berupa foto
    
    #untuk mendapatkan jumlah total gambar
    total_images = 0
    shape = None
    for images in glob.glob(dataset_path + '/**', recursive=True):
        if images[-3:] == 'pgm' or images[-3:] == 'jpg':
            total_images += 1
    
    shape = (112,92) #ukuran gambar
    all_images = np.zeros((total_images, shape[0], shape[1]) ,dtype='float64') #inisialisasi array pada numpy
    names = list()
    i = 0
    for folder in glob.glob(dataset_path + '/*'): #iterate through all the class
        for _ in range(100): #makes 10 copy of each class name in the list (since we have 10 images in each class)
            names.append(folder[-3:].replace('/', '')) #list for the classes of the faces
        for image in glob.glob(folder + '/*'): #iterate through each folder (class)
            read_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE) #read the image in grayscale
            resized_image = cv2.resize(read_image, (shape[1], shape[0])) #cv2.resize resizes an image into (# column x # height)
            all_images[i] = np.array(resized_image)
            i += 1
            
    A = np.resize(all_images, (total_images, shape[0]*shape[1])) #convert the images into vectors. Each row has an image vector. i.e. samples x image_vector matrix
    mean_vector = np.sum(A, axis=0, dtype='float64')/total_images #calculate the mean vector
    mean_matrix = np.tile(mean_vector, (total_images, 1)) #make a 400 copy of the same vector. 400 x image_vector_size matrix.
    A_tilde = A - mean_matrix #mean-subtracted image vectors
    
    L = (A_tilde.dot(A_tilde.T))/total_images #since each row is an image vector (unlike in the notes, L = (A_tilde)(A_tilde.T) instead of L = (A_tilde.T)(A_tilde)

    # print("L shape : ", L.shape)
    eigenvalues, eigenvectors = np.linalg.eig(L) #find the eigenvalues and the eigenvectors of L
    # print(eigenvalues)
    idx = eigenvalues.argsort()[::-1] #get the indices of the eigenvalues by its value. Descending order.
    eigenvalues = eigenvalues[idx] 
    eigenvectors = eigenvectors[:, idx] #sorted eigenvalues and eigenvectors in descending order
    
    eigenvectors_C = A_tilde.T @ eigenvectors #linear combination of each column of A_tilde
    eigenvectors_C.shape #each column is an eigenvector of C where C = (A_tilde.T)(A_tilde). NOTE : in the notes, C = (A_tilde)(A_tilde.T)
    
    #normalize the eigenvectors
    eigenfaces = preprocessing.normalize(eigenvectors_C.T) #normalize only accepts matrix with n_samples, n_feature. Hence the transpose.
    
    #to visualize some of the eigenfaces
    eigenface_labels = [x for x in range(eigenfaces.shape[0])] #list containing values from 1 to number of eigenfaces
    
    test_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) #testing image
    test_img = cv2.resize(test_img, (shape[1],shape[0])) #resize the testing image. cv2 resize by width and height.
    mean_subracted_testimg = np.reshape(test_img, (test_img.shape[0]*test_img.shape[1])) - mean_vector 
    
    q = 350 #number of chosen eigenfaces
    omega = eigenfaces[:q].dot(mean_subracted_testimg) #the vector that represents the image with respect to the eigenfaces.
    omega.shape
    
    #To visualize the reconstruction
    reconstructed = eigenfaces[:q].T.dot(omega) #image reconstructed using q eigenfaces.
    reconstructed.shape
    
    alpha_1 = 3000 #chosen threshold for face detection
    projected_new_img_vector = eigenfaces[:q].T @ omega #n^2 vector of the new face image represented as the linear combination of the chosen eigenfaces
    diff = mean_subracted_testimg - projected_new_img_vector 
    beta = math.sqrt(diff.dot(diff)) #distance between the original face image vector and the projected vector.
    
    alpha_2 = 3000 #chosen threshold for face recognition
    smallest_value = None #to keep track of the smallest value
    index = None #to keep track of the class that produces the smallest value
    for k in range(total_images):
        omega_k = eigenfaces[:q].dot(A_tilde[k]) #calculate the vectors of the images in the dataset and represent 
        diff = omega - omega_k
        epsilon_k = math.sqrt(diff.dot(diff))
        if smallest_value == None:
            smallest_value = epsilon_k
            index = k
        if smallest_value > epsilon_k:
            smallest_value = epsilon_k
            index = k
            
    if smallest_value < alpha_2:
        return jsonify(valuefinal=smallest_value, message="Success", statusCode=200, indexname=names[index]), 200
    
    else:
        return jsonify(valuefinal=smallest_value, message="Unknown Face!", statusCode=200), 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
