import argparse
import os
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inFolder", default="dresscode_json")
    parser.add_argument("--outFolder", default="cpvton_json_parsed")

    opt = parser.parse_args()
    return opt

def json_parser(json_elem):
    
    parsed_list = []
    
    data = {}
    data['version'] = 1.0,
    data['people'] = {}
    data['people']['face_keypoints'] = []
    data['people']['pose_keypoints'] = []
    data['people']['hand_right_keypoints'] = []
    data['people']['hand_left_keypoints'] = []
     

    keypoints = json_elem['keypoints']
        
    for keypoint in keypoints:
            
        if -1.0 not in keypoint[:3]: data['people']['pose_keypoints'].extend(keypoint[:3])
        else: data['people']['pose_keypoints'].extend([0,0,0])

        parsed_list.append(json.dumps(data))
        
    return json.dumps(data)



def main():
    args = get_args()
    outFolder = args.outFolder
    inFolder = args.inFolder
    
    listJson = []

    
    for filename in os.listdir(os.path.join(os.getcwd(), inFolder)):
        with open(os.path.join(os.path.join(os.getcwd(), inFolder), filename), 'r') as f: 
            listJson.append({'json':json.load(f), 'filename': filename})
        
    outJsons = []

    for elem in listJson:
        outJsons.append({'json':json_parser(elem['json']), 'filename': elem['filename'] })
        
    
    print(args)
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    
        print("Output directory created :'%s'" % (outFolder))
    else:
        print("Directory '%s' already exists" % (outFolder))
    
    for jselem in outJsons:
        with open(os.path.join(outFolder, jselem['filename']), 'w') as outfile:

            outfile.write(jselem['json'] + '\n')
        
 

if __name__ == "__main__":
    
    main()