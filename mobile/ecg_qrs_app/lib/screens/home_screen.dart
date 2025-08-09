import 'package:flutter/material.dart';
import 'package:ecg_qrs_app/widgets/labeled_logo.dart';
import 'package:ecg_qrs_app/screens/select_record_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final accent = Theme.of(context).colorScheme.primary;
    return Scaffold(
      appBar: AppBar(title: const Text('Detektor QRS')),
      body: SafeArea(
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(height: 40),
              LabeledLogo(isEnglish: false, size: 120),
              const SizedBox(height: 40),
              ElevatedButton.icon(
                icon: const Icon(Icons.input),
                label: const Text('Wpisz rekord (1–200)'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: accent,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                  textStyle: const TextStyle(fontSize: 18),
                ),
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const SelectRecordScreen()),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
