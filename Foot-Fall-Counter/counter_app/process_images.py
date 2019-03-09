import re
import os
import errno
from datetime import datetime
from PIL import Image

import requests
import cv2
import boto3

from api_exceptions import APIServiceUndefined

from conf import *


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

CV2_HASH_FILE_PATH = os.path.join(BASE_PATH, "cv2_hash_file.txt")
TO_PROCESS_FILE_PATH = os.path.join(BASE_PATH, "images_to_process.txt")
PROCESSED_FILE_PATH = os.path.join(BASE_PATH, "processed_images.txt")
HASH_FILE_PATH = os.path.join(BASE_PATH, "hashFile.txt")
TIMING_FILE_PATH = os.path.join(BASE_PATH, "Timing.txt")


# ##############PhotoHash Code#######################
def hash_distance(left_hash, right_hash):
    """Compute the hamming distance between two hashes"""
    if len(left_hash) != len(right_hash):
        raise ValueError('Hamming distance requires two strings of equal length')
    dist = sum(map(lambda x: 0 if -3 <= int(x[0], 16) - int(x[1], 16) <= 3 else 1, zip(left_hash, right_hash)))
    # print "\nDistance is: " + str(dist)
    return dist


def hashes_are_similar(left_hash, right_hash, tolerance=6):
    """
    Return True if the hamming distance between
    the image hashes are less than the given tolerance.
    """
    return hash_distance(left_hash, right_hash) <= tolerance


def average_hash(image_path, hash_size=8):
    """ Compute the average hash of the given image. """
    with open(image_path, 'rb') as f:
        # Open the image, resize it and convert it to black & white.
        image = Image.open(f).resize((hash_size, hash_size), Image.ANTIALIAS).convert('L')
        pixels = list(image.getdata())
    avg = sum(pixels) / len(pixels)

    # Compute the hash based on each pixels value compared to the average.
    bits = "".join(map(lambda pixel: '1' if pixel > avg +3 else '0', pixels))
    hashformat = "0{hashlength}x".format(hashlength=hash_size ** 2 // 4)
    return int(bits, 2).__format__(hashformat)


def make_sure_path_exists(path):
    """Function to ensure that a path exists before placing a file there. Creates a folder if it doesnt exist"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


if __name__ == "__main__":
    # Loading HaarCascade for Face Detection
    face_cascade = cv2.CascadeClassifier(os.path.join(BASE_PATH, 'haarcascade_frontalface_default.xml'))

    s3_client = boto3.client('s3', region_name=S3_CONF['region'])

    if FACE_API_SERVICE == AWS_SERVICE:
        # if AWS service is used, register the clients to use later
        rekognition_client = boto3.client('rekognition', region_name=REKOGNITION_CONF['region'])

        rekog_collection_name = STORE_CODE + "_" + datetime.now().strftime("%Y-%m-%d")
        # Create rekognition collection if it doesn't exist
        try:
            rekognition_client.create_collection(CollectionId=rekog_collection_name)
        except rekognition_client.exceptions.ResourceAlreadyExistsException:
            pass
    while True:
        f = open(TO_PROCESS_FILE_PATH, "r")
        to_process_images = set([x.strip() for x in f.readlines()])
        f.close()
        f = open(PROCESSED_FILE_PATH, "r")
        processed_images = set([x.strip() for x in f.readlines()])
        f.close()

        unprocessed_images = list(to_process_images - processed_images)

        for file_path in unprocessed_images:
            frame = cv2.imread(file_path)

            time_str = os.path.splitext(os.path.split(file_path)[-1])[0]
            date_str = os.path.split(os.path.split(file_path)[0])[-1]
            os.path.join(BASE_PATH, 'Photos', date_str, '{}.jpg'.format(time_str))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)  # Detect faces using Haas Cascades
            print "OPENCV:", faces
            if len(faces):
                print file_path

                face_count = 0
                face_det = False
                for face in faces:
                    (x, y, w, h) = face
                    face_path = os.path.join(BASE_PATH, 'Faces', date_str, '{}{}_cv2.jpg'.format(time_str, str(face_count)))
                    make_sure_path_exists(os.path.split(face_path)[0])
                    cv2.imwrite(face_path, frame[y:(y + h), x:(x + w)])
                    present_hash = average_hash(face_path)
                    f = open(CV2_HASH_FILE_PATH, "r")
                    is_old_face = False
                    for line in f:
                        line = line[:16]
                        is_old_face = hashes_are_similar(line, present_hash, 3)
                        if is_old_face:
                            os.remove(face_path)
                            break
                    f.close()
                    face_count += 1
                    if not is_old_face:
                        f = open(CV2_HASH_FILE_PATH, "a")
                        f.write(present_hash + "\n")
                        f.close()
                        face_det |= True
                if face_det:
                    try:
                        # Sends the image to aws/MS and gets the information
                        if FACE_API_SERVICE == MICROSOFT_SERVICE:
                            headers = {
                                'Content-Type': 'application/octet-stream',
                                'Ocp-Apim-Subscription-Key': MS_API_CONF['subscription_key'],
                            }
                            params = {
                                # Request parameters
                                'returnFaceId': 'true',
                                'returnFaceLandmarks': 'false',
                                'returnFaceAttributes': 'age,gender',
                            }

                            request_1 = requests.post(
                                MS_API_CONF['detect_url'],
                                headers=headers,
                                data=cv2.imencode('.jpg', frame)[1].tostring(),
                                params=params
                            )
                            response_1 = request_1.json()
                            response_faces = response_1
                        elif FACE_API_SERVICE == AWS_SERVICE:
                            response_1 = rekognition_client.index_faces(
                                CollectionId=rekog_collection_name,
                                Image={'Bytes': cv2.imencode('.jpg', frame)[1].tostring()},
                                DetectionAttributes=["ALL", ]
                            )
                            response_faces = response_1['FaceRecords']
                        else:
                            raise APIServiceUndefined()

                        print "Response1:", response_1
                        if not response_faces:
                            # If aws/MS finds zero faces, delete the file
                            try:
                                os.remove(file_path)
                                print "Wrong Detection"
                            except Exception as e:
                                print e
                        else:
                            # If aws/MS finds at least one face, save the file
                            fac = 0
                            # Flag to check if at least one new face exists in the picture.
                            # If not true, does not send to Google Analytics
                            for response_face in response_faces:
                                if FACE_API_SERVICE == MICROSOFT_SERVICE:
                                    face_dimensions = response_face['faceRectangle']
                                    x = face_dimensions['left']
                                    y = face_dimensions['top']
                                    w = face_dimensions['width']
                                    h = face_dimensions['height']
                                elif FACE_API_SERVICE == AWS_SERVICE:
                                    face_dimensions = response_face['Face']['BoundingBox']
                                    x = int(face_dimensions['Left'] * MAIN_IMAGE_DIMENSIONS['width'])
                                    y = int(face_dimensions['Top'] * MAIN_IMAGE_DIMENSIONS['height'])
                                    w = int(face_dimensions['Width'] * MAIN_IMAGE_DIMENSIONS['width'])
                                    h = int(face_dimensions['Height'] * MAIN_IMAGE_DIMENSIONS['height'])
                                else:
                                    raise APIServiceUndefined()
                                if w * h > DISTANCE_VALUE:
                                    face_path = os.path.join(BASE_PATH, 'Faces', date_str, '{}{}.jpg'.format(time_str, str(fac)))
                                    print face_path
                                    make_sure_path_exists(os.path.split(face_path)[0])
                                    cv2.imwrite(face_path, frame[y:(y + h), x:(x + w)])
                                    new_face = True
                                    if FACE_API_SERVICE == MICROSOFT_SERVICE:
                                        search_regex = "([0-9a-f-]*)," + date_str
                                        f = open(HASH_FILE_PATH, "r")
                                        existing_faces = re.findall(search_regex, f.read())
                                        f.close()
                                        if existing_faces:
                                            similar_face_headers = {'Ocp-Apim-Subscription-Key': MS_API_CONF['subscription_key']}
                                            similar_face_json = {
                                                'faceId': response_face['faceId'],
                                                'faceIds': existing_faces
                                            }

                                            similar_face_request = requests.post(
                                                MS_API_CONF['similar_face_url'], headers=similar_face_headers,
                                                json=similar_face_json)

                                            print "Response2:", similar_face_request.json()
                                            if similar_face_request.json():
                                                new_face = False
                                    elif FACE_API_SERVICE == AWS_SERVICE:
                                        rekog_search_response = rekognition_client.search_faces(
                                            CollectionId=rekog_collection_name,
                                            FaceId=response_face['Face']['FaceId']
                                        )
                                        print "Response2:", rekog_search_response
                                        if rekog_search_response['FaceMatches']:
                                            new_face = False
                                    else:
                                        raise APIServiceUndefined()

                                    if new_face:
                                        if FACE_API_SERVICE == MICROSOFT_SERVICE:
                                            f = open(HASH_FILE_PATH, "a")
                                            hashname = "{},{}\n".format(response_face['faceId'], date_str)
                                            f.write(hashname)
                                            f.close()

                                        print "New face found"
                                        # if new face found, save the faceId from MS API and send details to GA
                                        f = open(TIMING_FILE_PATH, "a")
                                        # gender = m for male and f for female
                                        if FACE_API_SERVICE == MICROSOFT_SERVICE:
                                            gender = response_face['faceAttributes']['gender'][0]
                                            age = int(round(response_face['faceAttributes']['age']))
                                            ga_cid = response_face['faceId']
                                        elif FACE_API_SERVICE == AWS_SERVICE:
                                            gender = response_face['FaceDetail']['Gender']['Value'][0]
                                            age = int(round(
                                                response_face['FaceDetail']['AgeRange']['Low'] +
                                                response_face['FaceDetail']['AgeRange']['High']
                                            ) / 2)
                                            ga_cid = response_face['Face']['FaceId']
                                        else:
                                            raise APIServiceUndefined()
                                        if age < 10:
                                            entry = "{0}^{1}^{2}^0{3}\n".format(date_str, time_str, gender, age)
                                        else:
                                            entry = "{0}^{1}^{2}^{3}\n".format(date_str, time_str, gender, age)
                                        print entry
                                        f.write(entry)
                                        f.close()

                                        ga_params = {
                                            'v': GA_API_CONF['version'],
                                            'tid': GA_API_CONF['tracking_id'],
                                            'cid': ga_cid,
                                            't': GA_API_CONF['hit_type'],
                                            'ec': STORE_CODE,
                                            'ea': "walk_in",
                                            'el': "{}_{}".format(gender, age),
                                            'ev': 1,
                                        }
                                        requests.get(GA_API_CONF['url'], params=ga_params)
                                        print "Face saved"
                                        s3_client.put_object(
                                            Body=open(face_path),
                                            Bucket=S3_CONF['face_bucket_path'],
                                            Metadata={
                                                'age': str(age),
                                                'gender': str(gender)
                                            },
                                            Key=STORE_CODE + "/" + date_str + "/" + os.path.split(face_path)[-1]
                                        )
                                    else:
                                        # probably remove this else part
                                        print "No new face"
                                else:
                                    print "Face dimensions are not large enough to be saved"

                                fac += 1

                    except Exception as e:
                        print e
                else:
                    "Duplicate found"
            else:
                os.remove(file_path)

            f = open(PROCESSED_FILE_PATH, "a")
            f.write(file_path + "\n")
            f.close()
