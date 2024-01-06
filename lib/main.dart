import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';

void main() {
  runApp(const TensorFlowApp());
}

class TensorFlowApp extends StatelessWidget {
  const TensorFlowApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Flutter TFLite',
      theme: ThemeData(
        primarySwatch: Colors.teal,
        appBarTheme: const AppBarTheme(elevation: 1),
      ),
      home: const TFLiteHome(),
    );
  }
}

class TFLiteHome extends StatefulWidget {
  const TFLiteHome({Key? key}) : super(key: key);

  @override
  _TFLiteHomeState createState() => _TFLiteHomeState();
}

const String ssd = 'SSD MobileNet';
const String yolo = 'Tiny YOLOv2';

class _TFLiteHomeState extends State<TFLiteHome> {
  String _model = ssd;
  File? _image;

  double? _imageWidth;
  double? _imageHeight;

  bool _busy = false;

  List? _recognitions;

  selectFromImagePicker() async {
    var imageRaw = await ImagePicker()
        .pickImage(source: ImageSource.gallery)
        .then((value) => value!.path);

    File? image = File(imageRaw);

    if (image == null) return;
    setState(() {
      _busy = true;
    });
    predictImage(image);
  }

  void predictImage(var image) async {
    if (image == null) return;

    if (_model == yolo) {
      await yolov5(image);
    } else {
      await ssdMobileNet(image);
    }

    FileImage(image)
        .resolve(const ImageConfiguration())
        .addListener(ImageStreamListener((imageInfo, _) {
      setState(() {
        _imageWidth = imageInfo.image.width.toDouble();
        _imageHeight = imageInfo.image.height.toDouble();
      });
    }));

    setState(() {
      _image = image;
      _busy = false;
    });
  }

  Future<void> yolov2(var image) async {
    var recognitions = await Tflite.detectObjectOnImage(
      path: image.path,
      model: "YOLO",
      threshold: 0.3,
      imageMean: 0.0,
      imageStd: 255.0,
      numResultsPerClass: 1,
    );

    adjustOutput(recognitions);
  }

  Future<void> ssdMobileNet(var image) async {
    var recognitions = await Tflite.detectObjectOnImage(
      path: image.path,
      numResultsPerClass: 1,
    );

    adjustOutput(recognitions);
  }

  void adjustOutput(List? recognitions) {
    if (recognitions == null) return;

    // Placeholder adjustment logic based on your specific requirements
    // This code adds zeros to each box to match the expected number of elements
    for (var recog in recognitions) {
      List<double> boxData = recog['rect'];
      List<double> adjustedBoxData = List.filled(4 - boxData.length, 0.0);
      adjustedBoxData.addAll(boxData);
      recog['rect'] = adjustedBoxData;
    }

    setState(() {
      _recognitions = recognitions;
    });
  }

  @override
  void initState() {
    super.initState();
    _busy = true;
    loadModel().then((val) {
      setState(() {
        _busy = false;
      });
    });
  }

  Future<void> loadModel() async {
    Tflite.close();
    try {
      String? res;
      if (_model == yolo) {
        res = await Tflite.loadModel(
          model: 'assets/best_float16.tflite',
          labels: 'assets/labels.txt',
        );
      } else {
        res = await Tflite.loadModel(
          model: 'assets/best_float16.tflite',
          labels: 'assets/labels.txt',
        );
      }
      print(res);
    } on PlatformException {
      print('Failed to load the model');
    }
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    List<Widget> stackChildren = [];

    stackChildren.add(
      Positioned(
        left: 0.0,
        top: 0.0,
        width: size.width,
        child: _image == null ? Text('No Image Selected') : Image.file(_image!),
      ),
    );

    stackChildren.addAll(renderBoxes(size));

    return Scaffold(
      appBar: AppBar(
        title: const Text('TensorFlow Lite Demo'),
        centerTitle: true,
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: selectFromImagePicker,
        tooltip: 'Select image form gallery',
        child: const Icon(Icons.image),
      ),
      body: Stack(
        children: stackChildren,
      ),
    );
  }

  List<Widget> renderBoxes(Size screen) {
    if (_recognitions == null) return [];

    if (_imageWidth == null || _imageHeight == null) return [];
    double factorX = screen.width;
    double factorY = _imageHeight! / _imageHeight! * screen.width;

    Color _blue = Colors.blue;
    return _recognitions!
        .map(
          (re) => Positioned(
        left: re['rect'][0] * factorX,
        top: re['rect'][1] * factorY,
        width: re['rect'][2] * factorX,
        height: re['rect'][3] * factorY,
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(5),
            border: Border.all(
              color: Colors.blue,
              width: 3,
            ),
          ),
          child: Text(
            "${re["detectedClass"]} ${(re["confidenceInClass"] * 100).toStringAsFixed(0)}%",
            style: TextStyle(background: Paint()..color = _blue),
          ),
        ),
      ),
    )
        .toList();
  }
}
