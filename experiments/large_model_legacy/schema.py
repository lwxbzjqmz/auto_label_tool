'''
large_model 返回JSON的协议

{
    "version":"1.0",
    "source":"large_model,
    "images":[
        {
            "image_name":"00000.jpg",
            "objects":[
                {
                    "name":"object",
                    "box":[xmin,ymin,xmax,ymax],
                    "score":0.95
                }
            ]
        }
    ]
}'''

LARGE_MODEL_VERSION="1.0"

LARGE_MODEL_SOURCE="large_model"

IMAGE_NAME_KEY="image_name"

OBJECTS_KEY="objects"

BOX_KEY="box"

NAME_KEY="name"

SCORE_KEY="score"