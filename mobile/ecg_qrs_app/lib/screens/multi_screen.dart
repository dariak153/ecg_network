import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:ecg_qrs_app/utils/constants.dart';

class MultiScreen extends StatefulWidget {
  final int recordId;
  const MultiScreen({Key? key, required this.recordId}) : super(key: key);

  @override
  State<MultiScreen> createState() => _MultiScreenState();
}

class _MultiScreenState extends State<MultiScreen> {
  final List<String> _leads = ['II', 'V1', 'V4', 'V5'];
  final Map<String, Uint8List?> _images = {};
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _fetchAllLeads();
  }

  Future<void> _fetchAllLeads() async {
    setState(() => _loading = true);
    for (var lead in _leads) {
      final uri = Uri.parse(
        '${Constants.apiUrl}/api/segment/record/${widget.recordId}/image'
        '?lead=$lead&start_sec=2&duration_sec=8',
      );
      final res = await http.get(uri);
      _images[lead] = res.statusCode == 200 ? res.bodyBytes : null;
    }
    setState(() => _loading = false);
  }

  @override
  Widget build(BuildContext context) {
    final accent = Theme.of(context).colorScheme.primary;
    return Scaffold(
      appBar: AppBar(
        title: Text('Porównanie odprowadzeń – rekord ${widget.recordId}'),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: _leads.map((lead) {
                  final bytes = _images[lead];
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Odprowadzenie $lead',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Container(
                          decoration: BoxDecoration(
                            border: Border.all(color: accent, width: 2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          padding: const EdgeInsets.all(4),
                          child: bytes != null
                              ? InteractiveViewer(
                                  maxScale: 4,
                                  child:
                                      Image.memory(bytes, fit: BoxFit.contain),
                                )
                              : const Center(
                                  child: Text(
                                    'Błąd ładowania obrazu',
                                    style: TextStyle(color: Colors.red),
                                  ),
                                ),
                        ),
                      ],
                    ),
                  );
                }).toList(),
              ),
            ),
    );
  }
}
