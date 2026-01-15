using UnityEngine;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Collections;
using Unity.InferenceEngine;
using UnityEngine.UI;
using System;
using TMPro;
using UnityEngine.Networking;
using System.IO.Compression;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class ImageSearchManager : MonoBehaviour
{
    [Header("Dependencies")]
    public ModelMobileCLIP2 mobileClipModel;
    public ScrollRect searchResultsViewPort;
    public GameObject imageResultPrefab;

    string imagesFolderName = "Images";
    string zipFileName = "Images.zip";

    [Header("Configuration")]
    public int TOP_K = 5;

    [Header("Debug")]
    public TMP_Text elapsedTimeText;

    private string _embeddingFilePath;
    private EmbeddingIndex _embeddingIndex;
    private Dictionary<string, Texture2D> _loadedTextures;
    private string _targetRootPath;
    private Coroutine _currentDisplayCoroutine;
    private bool _isSearching = false;

    void Awake()
    {
        _embeddingFilePath = Path.Combine(Application.streamingAssetsPath, "image_embeddings.bin");
        _loadedTextures = new Dictionary<string, Texture2D>();
    }

    IEnumerator Start()
    {
        yield return LoadEmbeddingIndexCoroutine();

#if UNITY_EDITOR
        SetupPathEditor();
#else
        yield return SetupPathAndroid();
#endif

        Debug.Log("System Ready");
    }

    private void SetupPathEditor()
    {
        _targetRootPath = Path.Combine(Application.streamingAssetsPath, imagesFolderName);
        Debug.Log($"Editor Image Root Path: {_targetRootPath}");
    }

    private IEnumerator SetupPathAndroid()
    {
        Debug.Log("Android Checking Image Files...");

        string sourceZipPath = Path.Combine(Application.streamingAssetsPath, zipFileName);
        string zipDestPath = Path.Combine(Application.persistentDataPath, zipFileName);
        string targetDir = Path.Combine(Application.persistentDataPath, imagesFolderName);

        bool needExtract = true;
        if (Directory.Exists(targetDir))
        {
            string[] existingFiles = Directory.GetFiles(targetDir, "*.*", SearchOption.AllDirectories);
            if (existingFiles.Length > 0)
            {
                Debug.Log($"Images found {existingFiles.Length} files. Skipping extraction.");
                needExtract = false;
            }
        }

        if (needExtract)
        {
            Debug.Log("Start Initialization...");

            if (Directory.Exists(targetDir))
            {
                Debug.Log("Deleting incomplete or old directory...");
                Directory.Delete(targetDir, true);
            }

            if (File.Exists(zipDestPath)) File.Delete(zipDestPath);

            Debug.Log($"Copying Zip from : {sourceZipPath}");

            using (UnityWebRequest www = UnityWebRequest.Get(sourceZipPath))
            {
                yield return www.SendWebRequest();

                if (www.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogError($"Zip Copy Failed: {www.error} URL: {sourceZipPath}");
                    yield break;
                }

                File.WriteAllBytes(zipDestPath, www.downloadHandler.data);
                Debug.Log("Zip Copy Success");
            }

            Debug.Log("Extracting Zip...");
            try
            {
                ZipFile.ExtractToDirectory(zipDestPath, Application.persistentDataPath);
                Debug.Log("Extraction Complete");
            }
            catch (Exception e)
            {
                Debug.LogError($"Extract Error: {e.Message} {e.StackTrace}");
            }
            finally
            {
                if (File.Exists(zipDestPath)) File.Delete(zipDestPath);
            }
        }

        _targetRootPath = targetDir;

        if (_embeddingIndex != null && _embeddingIndex.indexList.Count > 0)
        {
            string firstImageName = _embeddingIndex.indexList[0].imageName;
            string testPath = Path.Combine(_targetRootPath, firstImageName);

            if (!File.Exists(testPath))
            {
                string nestedPath = Path.Combine(_targetRootPath, imagesFolderName);
                if (Directory.Exists(nestedPath))
                {
                    Debug.Log("Nested folder detected. Adjusting root path.");
                    _targetRootPath = nestedPath;
                }
            }
        }

        Debug.Log($"Android Image Root Path Set: {_targetRootPath}");
    }

    IEnumerator LoadEmbeddingIndexCoroutine()
    {
        byte[] fileData = null;

        if (_embeddingFilePath.Contains("://") || _embeddingFilePath.Contains("jar:file"))
        {
            using (UnityWebRequest www = UnityWebRequest.Get(_embeddingFilePath))
            {
                yield return www.SendWebRequest();
                if (www.result == UnityWebRequest.Result.Success)
                    fileData = www.downloadHandler.data;
            }
        }
        else
        {
            if (File.Exists(_embeddingFilePath))
                fileData = File.ReadAllBytes(_embeddingFilePath);
        }

        if (fileData != null)
        {
            try
            {
                using (MemoryStream ms = new MemoryStream(fileData))
                using (BinaryReader reader = new BinaryReader(ms))
                {
                    int count = reader.ReadInt32();
                    _embeddingIndex = new EmbeddingIndex { indexList = new List<EmbeddingData>(count) };

                    for (int i = 0; i < count; i++)
                    {
                        string name = reader.ReadString();
                        int len = reader.ReadInt32();
                        float[] emb = new float[len];
                        for (int j = 0; j < len; j++) emb[j] = reader.ReadSingle();
                        _embeddingIndex.indexList.Add(new EmbeddingData { imageName = name, embedding = emb });
                    }
                }
                Debug.Log($"DB Loaded {_embeddingIndex.indexList.Count} embeddings.");
            }
            catch (Exception e)
            {
                Debug.LogError($"DB Parse failed: {e.Message}");
            }
        }
    }

    public void SearchTextToImage(TMP_InputField inputField)
    {
        if (string.IsNullOrWhiteSpace(inputField.text)) return;
        SearchTextToImage(inputField.text);
    }

    public async void SearchTextToImage(string queryText)
    {
        if (_embeddingIndex == null || _embeddingIndex.indexList.Count == 0) return;
        if (_isSearching) return;

        _isSearching = true;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            using (Tensor<float> textEmb = await mobileClipModel.GetTextEmbedding(queryText))
            {
                var results = await mobileClipModel.CalculateSimilarity(textEmb, _embeddingIndex);
                sw.Stop();
                StartDisplayCoroutine(results);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Search Error: {e.Message}");
        }
        finally
        {
            if (elapsedTimeText) elapsedTimeText.text = $"Search Time: {sw.ElapsedMilliseconds} ms\n({_embeddingIndex.indexList.Count} images)";
            _isSearching = false;
        }
    }

    public async void SearchImageToImage(Texture2D sourceImage)
    {
        if (_embeddingIndex == null || _embeddingIndex.indexList.Count == 0) return;
        if (_isSearching) return;

        _isSearching = true;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            using (Tensor<float> imgEmb = await mobileClipModel.GetImageEmbedding(sourceImage))
            {
                var results = await mobileClipModel.CalculateSimilarity(imgEmb, _embeddingIndex);
                sw.Stop();
                StartDisplayCoroutine(results);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Search Error: {e.Message}");
        }
        finally
        {
            if (elapsedTimeText) elapsedTimeText.text = $"Search Time: {sw.ElapsedMilliseconds} ms\n({_embeddingIndex.indexList.Count} images)";
            _isSearching = false;
        }
    }

    private void StartDisplayCoroutine(List<SimilarityResult> results)
    {
        if (_currentDisplayCoroutine != null) StopCoroutine(_currentDisplayCoroutine);
        _currentDisplayCoroutine = StartCoroutine(DisplayResultsLazy(results));
    }

    IEnumerator DisplayResultsLazy(List<SimilarityResult> results)
    {
        foreach (Transform child in searchResultsViewPort.content) Destroy(child.gameObject);

        var topResults = results.OrderByDescending(r => r.score).Take(TOP_K).ToList();

        foreach (var res in topResults)
        {
            Texture2D tex = null;

            if (_loadedTextures.TryGetValue(res.imageName, out Texture2D cachedTex))
            {
                tex = cachedTex;
            }
            else
            {
                string fullPath = Path.Combine(_targetRootPath, res.imageName);
                string fileUrl = "file://" + fullPath;

                using (UnityWebRequest www = UnityWebRequestTexture.GetTexture(fileUrl))
                {
                    yield return www.SendWebRequest();

                    if (www.result == UnityWebRequest.Result.Success)
                    {
                        tex = DownloadHandlerTexture.GetContent(www);
                        if (tex != null)
                        {
                            tex.name = res.imageName;
                            _loadedTextures[res.imageName] = tex;
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"Failed to load image: {res.imageName} {www.error}");
                    }
                }
            }

            if (tex != null)
            {
                GameObject obj = Instantiate(imageResultPrefab, searchResultsViewPort.content);

                if (obj.GetComponent<Image>())
                    obj.GetComponent<Image>().sprite = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), Vector2.one * 0.5f);
                else if (obj.GetComponent<RawImage>())
                    obj.GetComponent<RawImage>().texture = tex;

                var txt = obj.GetComponentInChildren<TMP_Text>();
                if (txt) txt.text = $"{res.score:F2}";
            }
        }
    }

    [ContextMenu("Generate DB Editor Mode")]
    public async void GenerateAndSaveEmbeddings()
    {
#if UNITY_EDITOR
        _embeddingFilePath = Path.Combine(Application.streamingAssetsPath, "image_embeddings.bin");
        string fullImagesPath = Path.Combine(Application.streamingAssetsPath, imagesFolderName);

        if (mobileClipModel == null || !Directory.Exists(fullImagesPath))
        {
            Debug.LogError("Model or Folder missing.");
            return;
        }

        mobileClipModel.InitializeForEditor();
        _embeddingIndex = new EmbeddingIndex { indexList = new List<EmbeddingData>() };

        string[] imageFiles = Directory.GetFiles(fullImagesPath, "*.*")
            .Where(s => s.EndsWith(".jpg") || s.EndsWith(".png")).ToArray();

        int count = 0;
        try
        {
            for (int i = 0; i < imageFiles.Length; i++)
            {
                string filePath = imageFiles[i];
                string fileName = Path.GetFileName(filePath);

                if (EditorUtility.DisplayCancelableProgressBar("Generating DB", fileName, (float)i / imageFiles.Length)) break;

                byte[] bytes = File.ReadAllBytes(filePath);
                Texture2D texture = new Texture2D(2, 2);
                texture.LoadImage(bytes);

                using (Tensor<float> embeddingTensor = await mobileClipModel.GetImageEmbedding(texture))
                {
                    if (embeddingTensor != null)
                    {
                        _embeddingIndex.indexList.Add(new EmbeddingData
                        {
                            imageName = fileName,
                            embedding = embeddingTensor.DownloadToArray()
                        });
                    }
                }
                DestroyImmediate(texture);
                count++;
            }
            SaveEmbeddingFile(count);
        }
        finally { EditorUtility.ClearProgressBar(); }
#endif
    }

    private void SaveEmbeddingFile(int count)
    {
        using (FileStream fs = new FileStream(_embeddingFilePath, FileMode.Create))
        using (BinaryWriter writer = new BinaryWriter(fs))
        {
            writer.Write(count);
            foreach (var data in _embeddingIndex.indexList)
            {
                writer.Write(data.imageName);
                writer.Write(data.embedding.Length);
                foreach (float value in data.embedding) writer.Write(value);
            }
        }
        Debug.Log($"Saved {count} embeddings.");
    }
}