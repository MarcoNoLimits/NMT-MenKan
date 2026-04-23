using System;
using System.Runtime.InteropServices;
using UnityEngine;
using System.IO;

public class HoloLensTranslator : MonoBehaviour
{
    // The name of our compiled C++ DLL (without the .dll extension)
    // When deploying to HoloLens, the DLL must be placed in Assets/Plugins/ARM64
    private const string PluginName = "NMT_MenKan_Plugin";

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void InitModel(string model_path);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr TranslateText(string input_text);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void CleanupModel();

    private bool isInitialized = false;

    void Start()
    {
        // For HoloLens, you should place your "en_it_v4_casual_weighted" folder inside the Unity StreamingAssets folder.
        // This ensures the files are copied to the headset during the build.
        string modelPath = Path.Combine(Application.streamingAssetsPath, "en_it_v4_casual_weighted");
        
        Debug.Log($"[NMT] Initializing Neural Machine Translation model at {modelPath}...");
        
        try {
            InitModel(modelPath);
            isInitialized = true;
            Debug.Log("[NMT] Model successfully initialized on ARM64 processor!");
        } 
        catch (Exception e) 
        {
            Debug.LogError($"[NMT] Failed to load native plugin or model: {e.Message}");
        }
    }

    /// <summary>
    /// Translates English text to Italian locally using the embedded NMT engine.
    /// </summary>
    public string Translate(string englishText)
    {
        if (!isInitialized)
        {
            Debug.LogWarning("[NMT] Cannot translate, model is not initialized.");
            return "Model not ready.";
        }

        // Call the C++ Native Plugin
        IntPtr resultPtr = TranslateText(englishText);
        
        // Convert the returned C string pointer back into a C# managed string
        string translatedItalian = Marshal.PtrToStringAnsi(resultPtr);
        
        return translatedItalian;
    }

    void OnDestroy()
    {
        if (isInitialized)
        {
            Debug.Log("[NMT] Cleaning up native translation resources.");
            CleanupModel();
            isInitialized = false;
        }
    }
}
