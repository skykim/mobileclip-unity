using UnityEngine;
using System.IO;
using System.Linq;
using System;
using Unity.InferenceEngine;
using FF = Unity.InferenceEngine.Functional;
using System.Collections.Generic;
using System.Collections;
using UnityEngine.Networking;
using System.Threading.Tasks;

public class ModelMobileCLIP2 : MonoBehaviour
{
    [Header("MobileCLIP2 Models (ONNX)")]
    public ModelAsset textEncoderAsset;
    public ModelAsset visionEncoderAsset;

    [Header("Configuration Files")]
    public string tokenizerJsonName = "tokenizer.json";

    private Worker _textWorker;
    private Worker _visionWorker;
    private Worker _dotScoreWorker;

    private BPETokenizer _tokenizer;

    private const BackendType BACKEND = BackendType.GPUCompute;
    private const int IMAGE_SIZE = 256;
    private const int CONTEXT_LENGTH = 77;
    private const int FEATURE_DIM = 512;

    IEnumerator Start()
    {
        yield return Initialize();
        RunWarmup();
    }

    void OnDestroy()
    {
        DisposeWorkers();
    }

    private void DisposeWorkers()
    {
        _textWorker?.Dispose();
        _visionWorker?.Dispose();
        _dotScoreWorker?.Dispose();
        _textWorker = null;
        _visionWorker = null;
        _dotScoreWorker = null;
    }

    public void InitializeForEditor()
    {
        if (_visionWorker != null && _textWorker != null) return;

        Debug.Log("Initializing MobileCLIP for Editor...");

        string path = Path.Combine(Application.streamingAssetsPath, tokenizerJsonName);
        if (!File.Exists(path))
        {
            Debug.LogError($"Tokenizer file not found at: {path}");
            return;
        }

        try
        {
            string jsonContent = File.ReadAllText(path);
            _tokenizer = new BPETokenizer(jsonContent);
        }
        catch (Exception e)
        {
            Debug.LogError($"Tokenizer Init Failed: {e.Message}");
            return;
        }

        LoadModelsAndWorkers();
    }

    private IEnumerator Initialize()
    {
        if (_visionWorker != null && _textWorker != null) yield break;

        string tokenizerJsonContent = "";
        yield return ReadStreamingAsset(tokenizerJsonName, (content) => tokenizerJsonContent = content);

        if (string.IsNullOrEmpty(tokenizerJsonContent)) yield break;

        try { _tokenizer = new BPETokenizer(tokenizerJsonContent); }
        catch (Exception e) { Debug.LogError(e); yield break; }

        LoadModelsAndWorkers();
    }

    private void LoadModelsAndWorkers()
    {
        try
        {
            if (_textWorker == null)
            {
                Model textModel = ModelLoader.Load(textEncoderAsset);
                _textWorker = new Worker(textModel, BACKEND);
            }

            if (_visionWorker == null)
            {
                Model visionModel = ModelLoader.Load(visionEncoderAsset);
                _visionWorker = new Worker(visionModel, BACKEND);
            }

            if (_dotScoreWorker == null)
            {
                FunctionalGraph graph = new FunctionalGraph();
                var inputA = graph.AddInput<float>(new TensorShape(1, FEATURE_DIM));
                var inputB = graph.AddInput<float>(new DynamicTensorShape(-1, FEATURE_DIM));

                var epsilon = FF.Constant(1e-12f);
                var normA = FF.Sqrt(FF.ReduceSum(inputA * inputA, 1, true));
                var normB = FF.Sqrt(FF.ReduceSum(inputB * inputB, 1, true));

                var vectorA = inputA / (normA + epsilon);
                var vectorB = inputB / (normB + epsilon);

                var sim = FF.ReduceSum(vectorA * vectorB, 1);
                Model dotModel = graph.Compile(sim);
                _dotScoreWorker = new Worker(dotModel, BACKEND);
            }

            Debug.Log("MobileCLIP Workers Ready.");
        }
        catch (Exception e)
        {
            Debug.LogError($"Worker Init Failed: {e.Message}");
        }
    }

    private IEnumerator ReadStreamingAsset(string fileName, Action<string> callback)
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        if (path.Contains("://") || path.Contains("jar:file"))
        {
            using (UnityWebRequest www = UnityWebRequest.Get(path))
            {
                yield return www.SendWebRequest();
                if (www.result == UnityWebRequest.Result.Success) callback(www.downloadHandler.text);
            }
        }
        else
        {
            if (File.Exists(path)) callback(File.ReadAllText(path));
        }
    }

    private async void RunWarmup()
    {
        if (_tokenizer == null) return;
        try
        {
            using var t = await GetTextEmbedding("warmup");
            Texture2D dummyTex = new Texture2D(IMAGE_SIZE, IMAGE_SIZE);
            using var i = await GetImageEmbedding(dummyTex);
            Destroy(dummyTex);
        }
        catch { }
    }

    public async Task<Tensor<float>> GetTextEmbedding(string text)
    {
        if (_tokenizer == null) return null;
        int[] tokenIds = _tokenizer.Encode(text, CONTEXT_LENGTH);
        using Tensor<int> inputTensor = new Tensor<int>(new TensorShape(1, CONTEXT_LENGTH), tokenIds);
        _textWorker.SetInput("input_ids", inputTensor);
        _textWorker.Schedule();
        var output = _textWorker.PeekOutput("text_embeds") as Tensor<float>;
        return await output.ReadbackAndCloneAsync();
    }

    public async Task<Tensor<float>> GetImageEmbedding(Texture image)
    {
        if (_visionWorker == null) return null;

        TextureTransform transform = new TextureTransform();
        transform.SetDimensions(width: IMAGE_SIZE, height: IMAGE_SIZE, channels: 3);
        transform.SetTensorLayout(TensorLayout.NCHW);

        using (Tensor<float> inputTensor = TextureConverter.ToTensor(image, transform))
        {
            _visionWorker.SetInput("pixel_values", inputTensor);
            _visionWorker.Schedule();

            var output = _visionWorker.PeekOutput("image_embeds") as Tensor<float>;
            Tensor<float> result = await output.ReadbackAndCloneAsync();
            return result;
        }
    }

    public async Task<List<SimilarityResult>> CalculateSimilarity(Tensor<float> queryEmbedding, EmbeddingIndex index)
    {
        if (index == null || index.indexList.Count == 0) return new List<SimilarityResult>();

        int N = index.indexList.Count;
        float[] galleryData = new float[N * FEATURE_DIM];
        int dimSize = FEATURE_DIM * sizeof(float);

        await Task.Run(() => {
            for (int i = 0; i < N; i++)
                Buffer.BlockCopy(index.indexList[i].embedding, 0, galleryData, i * dimSize, dimSize);
        });

        using (Tensor<float> galleryTensor = new Tensor<float>(new TensorShape(N, FEATURE_DIM), galleryData))
        {
            _dotScoreWorker.SetInput(0, queryEmbedding);
            _dotScoreWorker.SetInput(1, galleryTensor);
            _dotScoreWorker.Schedule();

            var output = _dotScoreWorker.PeekOutput() as Tensor<float>;
            using var cpuOutput = await output.ReadbackAndCloneAsync();
            float[] scores = cpuOutput.DownloadToArray();

            List<SimilarityResult> results = new List<SimilarityResult>();
            for (int i = 0; i < N; i++) results.Add(new SimilarityResult { imageName = index.indexList[i].imageName, score = scores[i] });
            return results;
        }
    }
}