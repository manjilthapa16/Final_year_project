import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:webview_flutter_android/webview_flutter_android.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Hide status bar + nav bar for a true full-screen camera experience.
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  runApp(const GymBuddyApp());
}

class GymBuddyApp extends StatelessWidget {
  const GymBuddyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Your Gym Buddy',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: const ColorScheme.dark(),
      ),
      home: const WebViewPage(),
    );
  }
}

class WebViewPage extends StatefulWidget {
  const WebViewPage({super.key});

  @override
  State<WebViewPage> createState() => _WebViewPageState();
}

class _WebViewPageState extends State<WebViewPage> {
  late final WebViewController _controller;
  late final FlutterTts _tts;
  bool _ready = false;

  @override
  void initState() {
    super.initState();
    _initTts();
    _initWebView();
  }

  Future<void> _initTts() async {
    _tts = FlutterTts();
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.5);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
  }

  Future<void> _initWebView() async {
    // Request camera permission at the native layer (Android 6+).
    if (defaultTargetPlatform == TargetPlatform.android) {
      await Permission.camera.request();
    }

    // Create the navigation delegate first so we can configure the Android-
    // specific SSL-error callback on it before attaching it to the controller.
    final navDelegate = NavigationDelegate(
      onPageFinished: (_) {
        if (mounted) setState(() => _ready = true);
      },
      onWebResourceError: (_) {
        if (mounted && !_ready) setState(() => _ready = true);
      },
    );

    // Trust the self-signed HTTPS cert from the Vite dev server so the page
    // loads as a SECURE CONTEXT — Android WebView blocks getUserMedia on any
    // plain HTTP origin that is not localhost.
    if (defaultTargetPlatform == TargetPlatform.android) {
      final AndroidNavigationDelegate androidDelegate =
          navDelegate.platform as AndroidNavigationDelegate;
      await androidDelegate.setOnSSlAuthError((error) async {
        await error.proceed();
      });
    }

    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xFF0a0a0a))
      ..setNavigationDelegate(navDelegate)
      ..addJavaScriptChannel(
        'NativeTts',
        onMessageReceived: (JavaScriptMessage msg) {
          final text = msg.message.trim();
          if (text.isNotEmpty) {
            _tts.stop().then((_) => _tts.speak(text));
          }
        },
      );

    // Android: configure media & camera permissions BEFORE loading the page
    // to avoid a race condition where a permission request fires before the
    // handler is registered.
    if (defaultTargetPlatform == TargetPlatform.android) {
      final AndroidWebViewController androidController =
          _controller.platform as AndroidWebViewController;
      // Allow the <video> element to autoplay without a user gesture.
      await androidController.setMediaPlaybackRequiresUserGesture(false);
      await androidController.setOnPlatformPermissionRequest((
        PlatformWebViewPermissionRequest request,
      ) {
        request.grant();
      });
    }

    await _controller.loadRequest(Uri.parse('${_webUrl()}?platform=mobile'));
  }

  String _webUrl() {
    if (kIsWeb) return 'http://localhost:5173';
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return 'https://192.168.1.11:5173';
      default:
        return 'http://localhost:5173';
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_ready) {
      return const Scaffold(
        backgroundColor: Color(0xFF0a0a0a),
        body: Center(child: CircularProgressIndicator(color: Colors.white)),
      );
    }
    return Scaffold(
      backgroundColor: const Color(0xFF0a0a0a),
      body: SafeArea(child: WebViewWidget(controller: _controller)),
    );
  }
}
