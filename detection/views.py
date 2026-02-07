from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .ml.detector import process_image, process_video
from .ml.ocr import extract_text
from django.http import StreamingHttpResponse
from .ml.detector import webcam_stream



def upload_media(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_name = uploaded_file.name.lower()

        upload_dir = settings.MEDIA_ROOT / 'uploads'
        result_dir = settings.MEDIA_ROOT / 'results'
        fs = FileSystemStorage(location=upload_dir)

        filename = fs.save(uploaded_file.name, uploaded_file)
        input_path = fs.path(filename)

        # ---------- IMAGE ----------
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            output_path = result_dir / filename

            helmet_status, _ = process_image(input_path, str(output_path))
            extracted_text = extract_text(input_path)

            context = {
                'type': 'image',
                'helmet_status': helmet_status,
                'extracted_text': extracted_text,
                'result_media': settings.MEDIA_URL + 'results/' + filename
            }

        # ---------- VIDEO ----------
        elif file_name.endswith(('.mp4', '.avi', '.mov')):
            output_video = "processed_" + filename
            output_path = result_dir / output_video

            stats = process_video(input_path, str(output_path))

            context = {
                'type': 'video',
                'result_media': settings.MEDIA_URL + 'results/' + output_video,
                'stats': stats
            }

    return render(request, 'detection/index.html', context)
def webcam_view(request):
    return render(request, 'detection/webcam.html')


def webcam_feed(request):
    return StreamingHttpResponse(
        webcam_stream(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
