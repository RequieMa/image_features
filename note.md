# Steps to Use ONNXRuntime in Dart
## Add the Dependency: First, add the onnxruntime package to your pubspec.yaml file.
```yaml
dependencies:
  onnxruntime: ^x.y.z
```
## Import the Package: Import the onnxruntime package in your Dart code.
```dart
import 'package:onnxruntime/onnxruntime.dart';
```
## Initialize the Environment: Initialize the ONNX Runtime environment. This is typically done once at the start of your application.
```dart
void main() {
  OrtEnv.instance.init();
  // Your app initialization code
}
```
## Create a Session: Load your ONNX model and create a session. You can load the model from assets or a file.
```dart 
import 'package:flutter/services.dart' show rootBundle;

Future<OrtSession> createSession() async {
  final sessionOptions = OrtSessionOptions();
  const assetFileName = 'assets/models/test.onnx';
  final rawAssetFile = await rootBundle.load(assetFileName);
  final bytes = rawAssetFile.buffer.asUint8List();
  return OrtSession.fromBuffer(bytes, sessionOptions);
}
```
## Perform Inference: Prepare the input data, run the inference, and handle the output.
```dart
Future<void> runInference(OrtSession session) async {
  final shape = [1, 2, 3]; // Example shape
  final data = [1.0, 2.0, 3.0]; // Example data
  final inputOrt = OrtValueTensor.createTensorWithDataList(data, shape);
  final inputs = {'input': inputOrt};
  final runOptions = OrtRunOptions();
  final outputs = await session.runAsync(runOptions, inputs);

  // Process the outputs
  outputs?.forEach((element) {
    print(element?.data);
    element?.release();
  });

  inputOrt.release();
  runOptions.release();
}
```
## Release the Environment: When your application is done using the ONNX Runtime, release the environment.
```dart
void dispose() {
  OrtEnv.instance.release();
}
```
# Example Code
Here’s a complete example that ties everything together:
```dart
import 'package:flutter/material.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:flutter/services.dart' show rootBundle;

void main() {
  OrtEnv.instance.init();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('ONNX Runtime Example')),
        body: Center(
          child: FutureBuilder<OrtSession>(
            future: createSession(),
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                if (snapshot.hasError) {
                  return Text('Error: ${snapshot.error}');
                }
                final session = snapshot.data!;
                return ElevatedButton(
                  onPressed: () => runInference(session),
                  child: Text('Run Inference'),
                );
              } else {
                return CircularProgressIndicator();
              }
            },
          ),
        ),
      ),
    );
  }

  Future<OrtSession> createSession() async {
    final sessionOptions = OrtSessionOptions();
    const assetFileName = 'assets/models/test.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    return OrtSession.fromBuffer(bytes, sessionOptions);
  }

  Future<void> runInference(OrtSession session) async {
    final shape = [1, 2, 3]; // Example shape
    final data = [1.0, 2.0, 3.0]; // Example data
    final inputOrt = OrtValueTensor.createTensorWithDataList(data, shape);
    final inputs = {'input': inputOrt};
    final runOptions = OrtRunOptions();
    final outputs = await session.runAsync(runOptions, inputs);

    // Process the outputs
    outputs?.forEach((element) {
      print(element?.data);
      element?.release();
    });

    inputOrt.release();
    runOptions.release();
  }
}
```
This example demonstrates how to set up and use the ONNXRuntime package in a Flutter application.

# more example
Notice that none of the generated example is good enough

```dart
class MyModel {
  OrtSession? _session;

  Future<void> loadModel(Uint8List modelData) async {
    final sessionOptions = OrtSessionOptions();
    _session = await OrtSession.fromBuffer(modelData, sessionOptions);
  }

  Future<void> runInference() async {
    if (_session == null) return;
    final shape = [1, 2, 3]; // Example shape
    final data = [1.0, 2.0, 3.0]; // Example data
    final inputOrt = OrtValueTensor.createTensorWithDataList(data, shape);
    final inputs = {'input': inputOrt};
    final runOptions = OrtRunOptions();
    final outputs = await _session!.runAsync(runOptions, inputs);

    // Process the outputs
    outputs?.forEach((element) {
      print(element?.data);
      element?.release();
    });

    inputOrt.release();
    runOptions.release();
  }

  void dispose() {
    _session?.release();
  }
}

void main() {
  final model = MyModel();
  // Load model, run inference, etc.
  model.dispose(); // Clean up resources
}
```

Use `OrtSession.fromFile` if your model is stored locally and you prefer simplicity.
Use `OrtSession.fromBuffer` if your model data is dynamic or fetched from a network.

# more
```dart
import 'dart:io';
import 'dart:typed_data';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;

class ImageDuplicateFinder {
  late OrtSession _session;
  late OrtEnv _env;
  late OrtRunOptions _runOptions;
  late OrtSessionOptions _sessionOptions;

  ImageDuplicateFinder(String modelPath) {
    _env = OrtEnv.instance;
    _env.init();
    _sessionOptions = OrtSessionOptions();
    _session = OrtSession.fromFile(File(modelPath), _sessionOptions);
    _runOptions = OrtRunOptions();
  }

  Future<void> dispose() async {
    _session.release();
    _runOptions.release();
    _sessionOptions.release();
    _env.release();
  }

  Future<Uint8List> _processImage(File imageFile) async {
    final image = img.decodeImage(imageFile.readAsBytesSync())!;
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      image.getBytes().map((e) => e.toDouble()).toList(),
      [1, image.height, image.width, 3],
    );
    final inputs = {'input': inputTensor};
    final outputs = await _session.runAsync(_runOptions, inputs);
    inputTensor.release();
    return outputs!.first.data as Uint8List;
  }

  Future<void> setQueryImage(File queryImage) async {
    // Process the query image and store the result
    final queryResult = await _processImage(queryImage);
    // Store or use the queryResult as needed
  }

  Future<List<File>> findDuplicates(Directory folder) async {
    final duplicates = <File>[];
    final files = folder.listSync().whereType<File>();
    for (final file in files) {
      final result = await _processImage(file);
      // Compare result with queryResult to find duplicates
      // If duplicate, add to duplicates list
    }
    return duplicates;
  }
}

void main() async {
  final modelPath = 'path/to/your/model.onnx';
  final finder = ImageDuplicateFinder(modelPath);

  final queryImage = File('path/to/query/image.jpg');
  await finder.setQueryImage(queryImage);

  final folder = Directory('path/to/folder');
  final duplicates = await finder.findDuplicates(folder);

  print('Found duplicates: ${duplicates.map((f) => f.path).toList()}');

  await finder.dispose();
}
```
Explanation
1. Initialization:
    - The constructor initializes the ONNX Runtime environment and session.
    - The dispose method releases all resources.
2. High-Level Methods:
    - setQueryImage: Processes the query image and stores the result.
    - findDuplicates: Processes each image in the folder, compares it with the query image, and returns a list of duplicates.
3. ncapsulation:
    - The user interacts with high-level methods without worrying about the underlying initialization and release of resources.

This design ensures that the user has a simple and clean interface to work with, while the class handles all the complex details internally.

# Device Flags
```dart
final coreMLFlags = CoreMLFlags.useNone | CoreMLFlags.useCpuOnly | CoreMLFlags.enableOnSubgraph | CoreMLFlags.onlyEnableDeviceWithANE;
  sessionOptions.appendCoreMLProvider(coreMLFlags);

final sessionOptions = OrtSessionOptions();

// Append multiple providers
sessionOptions.appendCoreMLProvider(CoreMLFlags.useNone);
sessionOptions.appendCPUProvider(CPUFlags.useArena);
sessionOptions.appendNnapiProvider(NnapiFlags.useNone);
sessionOptions.appendXnnpackProvider();

final modelFile = File('path/to/your/model.onnx');
final session = await OrtSession.fromFile(modelFile, sessionOptions);
```