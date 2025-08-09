import 'package:flutter/material.dart';
import 'package:ecg_qrs_app/screens/segment_screen.dart';

class SelectRecordScreen extends StatefulWidget {
  const SelectRecordScreen({Key? key}) : super(key: key);

  @override
  State<SelectRecordScreen> createState() => _SelectRecordScreenState();
}

class _SelectRecordScreenState extends State<SelectRecordScreen> {
  final _controller = TextEditingController();
  String? _error;
  String _lead = 'II';
  final _leads = ['II', 'V1', 'V4', 'V5'];

  void _submit() {
    final id = int.tryParse(_controller.text.trim());
    if (id == null || id < 1 || id > 200) {
      setState(() => _error = 'Podaj liczbę od 1 do 200');
      return;
    }
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => SegmentScreen(recordId: id, initialLead: _lead),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final accent = Theme.of(context).colorScheme.primary;
    return Scaffold(
      appBar: AppBar(title: const Text('Wybór rekordu')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    keyboardType: TextInputType.number,
                    decoration: InputDecoration(
                      labelText: 'Numer rekordu',
                      errorText: _error,
                      border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8)),
                      focusedBorder: OutlineInputBorder(
                        borderSide: BorderSide(color: accent, width: 2),
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    onChanged: (_) {
                      if (_error != null) setState(() => _error = null);
                    },
                    onSubmitted: (_) => _submit(),
                  ),
                ),
                const SizedBox(width: 12),
                DropdownButton<String>(
                  value: _lead,
                  items: _leads
                      .map((l) =>
                          DropdownMenuItem(value: l, child: Text('Odp. $l')))
                      .toList(),
                  onChanged: (v) => setState(() => _lead = v!),
                ),
              ]),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _submit,
                style: ElevatedButton.styleFrom(
                  backgroundColor: accent,
                  padding:
                      const EdgeInsets.symmetric(vertical: 14, horizontal: 32),
                  textStyle: const TextStyle(fontSize: 16),
                ),
                child: const Text('Dalej'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
