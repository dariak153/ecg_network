import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:ecg_qrs_app/screens/home_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await dotenv.load(); // wczytanie .env
  runApp(const EcgApp());
}

class EcgApp extends StatelessWidget {
  const EcgApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    const primary = Color(0xFF0B91FF);
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Detektor QRS',
      theme: ThemeData(
        colorScheme: const ColorScheme.light(
          primary: primary,
          secondary: primary,
          onPrimary: Colors.white,
        ),
        appBarTheme: const AppBarTheme(backgroundColor: primary),
      ),
      home: const HomeScreen(),
    );
  }
}
