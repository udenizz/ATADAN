
using Atadan.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using static System.Collections.Specialized.BitVector32;
using Image = SixLabors.ImageSharp.Image;
using Size = SixLabors.ImageSharp.Size;

namespace Atadan.AiModel2 {
    public class AIModel2 {
        const int DimBatchSize = 1;
        const int DimNumberOfChannels = 3;
        const int ImageSizeX = 128;
        const int ImageSizeY = 128;
        const string ModelInputName = "input";
        const string ModelOutputName = "output";

        byte[] _model;
        byte[] _sampleImage;
        List<string> _labels;
        InferenceSession _session;
        Task _initTask;
        byte[] img;

        public AIModel2(byte[] image) {
            img = image;
            _ = InitAsync();
        }
        Task InitAsync() {
            if (_initTask == null || _initTask.IsFaulted)
                _initTask = InitTask();

            return _initTask;
        }

        public async Task<byte[]> GetSampleImageAsync() {
            await InitAsync().ConfigureAwait(false);
            return _sampleImage;
        }


        async Task InitTask() {
            var assembly = GetType().Assembly;

            var xd = assembly.GetName().Name;
            var assembly2 = Assembly.GetExecutingAssembly();
            var resourceNames = assembly2.GetManifestResourceNames();



            //// Get labels
            using var labelsStream = assembly.GetManifestResourceStream("Atadan.Assets.plant_diseases.txt");
            using var reader = new StreamReader(labelsStream);

            string text = await reader.ReadToEndAsync();
            _labels = text.Split(new string[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries).ToList();

            // Get model and create session
            using var modelStream = assembly.GetManifestResourceStream("Atadan.Assets.model.onnx");
            using var modelMemoryStream = new MemoryStream();

            modelStream.CopyTo(modelMemoryStream);
            _model = modelMemoryStream.ToArray();
            _session = new InferenceSession(_model);




            // Get sample image
            //using var sampleImageStream = assembly.GetManifestResourceStream("Atadan.Assets.test2.jpeg");
            //using var sampleImageMemoryStream = new MemoryStream();

            //sampleImageStream.CopyTo(sampleImageMemoryStream);
            _sampleImage = img.ToArray();
        }


        public async Task<AiResultModel> GetClassificationAsync(byte[] image) {
            await InitAsync().ConfigureAwait(false);
            var rszimg = ResizeImage(img);
            using var sourceBitmap = SKBitmap.Decode(rszimg);
            var pixels = sourceBitmap.Bytes;

            // Preprocess image data according to model requirements
            // https://github.com/onnx/models/tree/master/vision/classification/mobilenet#preprocessing

            // Scale and crop the original image if necessary to match the way the model has been trained
            // In this case, the model expects 224x224 images
            if (sourceBitmap.Width != ImageSizeX || sourceBitmap.Height != ImageSizeY) {
                float ratio = (float)Math.Min(ImageSizeX, ImageSizeY) / Math.Min(sourceBitmap.Width, sourceBitmap.Height);

                using SKBitmap scaledBitmap = sourceBitmap.Resize(new SKImageInfo(
                    (int)(ratio * sourceBitmap.Width),
                    (int)(ratio * sourceBitmap.Height)),
                    SKFilterQuality.Medium);

                var horizontalCrop = scaledBitmap.Width - ImageSizeX;
                var verticalCrop = scaledBitmap.Height - ImageSizeY;
                var leftOffset = horizontalCrop == 0 ? 0 : horizontalCrop / 2;
                var topOffset = verticalCrop == 0 ? 0 : verticalCrop / 2;

                var cropRect = SKRectI.Create(
                    new SKPointI(leftOffset, topOffset),
                    new SKSizeI(ImageSizeX, ImageSizeY));

                using SKImage currentImage = SKImage.FromBitmap(scaledBitmap);
                using SKImage croppedImage = currentImage.Subset(cropRect);
                using SKBitmap croppedBitmap = SKBitmap.FromImage(croppedImage);

                pixels = croppedBitmap.Bytes;
            }

            // Preprocess the image data into the format expected by the model.

            // In this case, the model expects RGB values in the range of [0, 1] normalized using
            // mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

            // The loop below iterates over the image pixels one row at a time,
            // applies the requisite normalization to each RGB value, then stores each in the channelData array.

            // The channelData array stores the normalized RGB values sequentially one channel at a time (instead of the original RGB, RGB, ... sequence) i.e.
            // first all the R values,
            // then all the G values,
            // then all the B values

            // The resulting channelData array is used to create the requisite Tensor object as input to the InferenceSession.Run method
//            var bytesPerPixel = sourceBitmap.BytesPerPixel;
//            var rowLength = ImageSizeX * bytesPerPixel;
//            var channelLength = ImageSizeX * ImageSizeY;
//            var channelData = new float[channelLength * 3];
//            var channelDataIndex = 0;
//;
//            for (int y = 0; y < ImageSizeY; y++) {
//                var rowOffset = y * rowLength;

//                for (int x = 0, columnOffset = 0; x < ImageSizeX; x++, columnOffset += bytesPerPixel) {
//                    var pixelOffset = rowOffset + columnOffset;

//                    var pixelR = pixels[pixelOffset];
//                    var pixelG = pixels[pixelOffset + 1];
//                    var pixelB = pixels[pixelOffset + 2];

//                    var rChannelIndex = channelDataIndex;
//                    var gChannelIndex = channelDataIndex + channelLength;
//                    var bChannelIndex = channelDataIndex + (channelLength * 2);

//                    channelData[rChannelIndex] = (pixelR / 255f);
//                    channelData[gChannelIndex] = (pixelG / 255f);
//                    channelData[bChannelIndex] = (pixelB / 255f);

//                    channelDataIndex++;
//                }
//            }
            Stream str = new MemoryStream(rszimg);
            Image<Rgba32> imagetemp = Image.Load<Rgba32>(str);
            int width = imagetemp.Width;
            int height = imagetemp.Height;

            float[] floatArray = new float[width * height * 3]; // 4 channels: RGBA

            int index = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Rgba32 pixel = imagetemp[x, y];
                    floatArray[index++] = pixel.R / 255f; // Red channel
                    floatArray[index++] = pixel.G / 255f; // Green channel
                    floatArray[index++] = pixel.B / 255f; // Blue channel
                }
            }



            // Create Tensor model input
            // The model expects input to be in the shape of (N x 3 x H x W) i.e.
            // mini-batches (where N is the batch size) of 3-channel RGB images with H and W of 224
            // https://onnxruntime.ai/docs/api/csharp-api#systemnumericstensor
            var input = new DenseTensor<float>(floatArray, new[] { DimBatchSize, ImageSizeX, ImageSizeY, DimNumberOfChannels });

            // Run inferencing
            // https://onnxruntime.ai/docs/api/csharp-api#methods-1
            // https://onnxruntime.ai/docs/api/csharp-api#namedonnxvalue
            //using var results = _session.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(ModelInputName, input) });
            var inputName = _session.InputMetadata.Keys.First();
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, input)
            };

            var results = _session.Run(inputs);
            // Resolve model output
            // https://github.com/onnx/models/tree/master/vision/classification/mobilenet#output
            // https://onnxruntime.ai/docs/api/csharp-api#disposablenamedonnxvalue
            var outputName = _session.OutputMetadata.Keys.First();
            var output = results.FirstOrDefault(i => i.Name == outputName);

            if (output == null)
                return null;

            // Postprocess output (get highest score and corresponding label)
            // https://github.com/onnx/models/tree/master/vision/classification/mobilenet#postprocessing
            var scores = output.AsTensor<float>().ToList();
            var highestScore = scores.Max();
            var highestScoreIndex = scores.IndexOf(highestScore);
            var label = _labels.ElementAt(highestScoreIndex);
            AiResultModel aiResultModel = new AiResultModel();
            aiResultModel.diseaseName = label;
            aiResultModel.correctRate = highestScore * 100;
            return aiResultModel;
        }
        byte[] ResizeImage(byte[] imageBytes) {
            using (Image image = Image.Load(imageBytes)) {
                int targetWidth = 128;
                int targetHeight = 128;

                image.Mutate(x => x.Resize(new ResizeOptions {
                    Size = new Size(targetWidth, targetHeight),
                    Mode = SixLabors.ImageSharp.Processing.ResizeMode.Stretch
                }));

                using (MemoryStream outputStream = new MemoryStream()) {
                    image.Save(outputStream, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                    return outputStream.ToArray();
                }
            }
        }
    }

}