import cloudinary.uploader
import cloudinary

cloudinary_info = {
    "cloud_name": "dhmozdnjd",
    "api_key": "772868216296527",
    "api_secret": "po1MBzLDpswneuZDDVaz1-DXc5g"
}

def cursor_to_list(cursor):
    my_list = []
    for obj in cursor:
        my_list.append(obj)
    return my_list

def upload_image(image):
    cloudinary.config(cloud_name=cloudinary_info['cloud_name'],
                      api_key=cloudinary_info['api_key'],
                      api_secret=cloudinary_info['api_secret'])
    upload_result = None
    image_to_upload = image
    if image_to_upload:
        upload_result = cloudinary.uploader.upload(image_to_upload)
        return upload_result


def transform_image(upload_result, width, height):
    t = cloudinary.utils.cloudinary_url(
        upload_result['public_id'] + '.' + upload_result['format'], width=width, height=height, crop="fit")
    return t


