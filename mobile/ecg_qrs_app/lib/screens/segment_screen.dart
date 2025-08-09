import 'dart:convert';
import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui' as ui;
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:signature/signature.dart';
import 'package:flutter_email_sender/flutter_email_sender.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:pdf/pdf.dart';
import 'package:ecg_qrs_app/utils/constants.dart';
import 'package:ecg_qrs_app/widgets/stat_card.dart';
import 'package:ecg_qrs_app/widgets/annotation_overlay.dart';
import 'package:ecg_qrs_app/screens/multi_screen.dart';

class SegmentScreen extends StatefulWidget {
  final int recordId;
  final String initialLead;
  const SegmentScreen({
    Key? key,
    required this.recordId,
    this.initialLead = 'II',
  }) : super(key: key);

  @override
  State<SegmentScreen> createState() => _SegmentScreenState();
}

class _SegmentScreenState extends State<SegmentScreen> {
  Uint8List? _img;
  bool _loading = true;
  String? _error;
  double? _avg, _min, _max, _hr;
  late String _lead;
  final _leads = ['II', 'V1', 'V4', 'V5'];
  bool _annotationMode = false;

  // PIN & password display
  bool _pinned = false;
  late final String _reportPassword;

  // electronic signature controller
  final SignatureController _sigController = SignatureController(
    penStrokeWidth: 2,
    penColor: Colors.black,
    exportBackgroundColor: Colors.white,
  );

  @override
  void initState() {
    super.initState();
    _lead = widget.initialLead;
    _reportPassword = _generatePassword();
    _fetchAll();
  }

  String _generatePassword() {
    final code = (widget.recordId * 123456) % 1000000;
    return code.toString().padLeft(6, '0');
  }

  Future<void> _fetchAll() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final base = Constants.apiUrl;
      final imgRes = await http.get(Uri.parse(
          '$base/api/segment/record/${widget.recordId}/image?lead=$_lead&start_sec=2&duration_sec=8'));
      if (imgRes.statusCode != 200) throw 'Obraz: ${imgRes.statusCode}';
      _img = imgRes.bodyBytes;

      final statsRes = await http.get(Uri.parse(
          '$base/api/segment/record/${widget.recordId}/stats?lead=$_lead&start_sec=2&duration_sec=8'));
      if (statsRes.statusCode != 200) throw 'Stats: ${statsRes.statusCode}';
      final js = jsonDecode(statsRes.body);
      _avg = js['average_qrs'];
      _min = js['min_qrs'];
      _max = js['max_qrs'];
      _hr = js['heart_rate'];
    } catch (e) {
      _error = e.toString();
    } finally {
      setState(() => _loading = false);
    }
  }

  String _formatMs(double sec) {
    final s = (sec * 1000).toStringAsFixed(3);
    return s.replaceAll(RegExp(r'\.?0+\$'), '') + ' ms';
  }

  String get _qrsClass {
    if (_avg == null) return '';
    final ms = _avg! * 1000;
    if (ms < 100) return 'QRS w normie (<100 ms)';
    if (ms < 120) return 'QRS lekko poszerzony (100–120 ms)';
    return 'QRS szeroki (≥120 ms)';
  }

  String get _hrClass {
    if (_hr == null) return '';
    if (_hr! < 60) return 'Bradykardia (<60 bpm)';
    if (_hr! > 100) return 'Tachykardia (>100 bpm)';
    return 'Norma (60–100 bpm)';
  }

  Future<File> _createPdfFile() async {
    final pdf = pw.Document();
    final fontData = await rootBundle.load('assets/fonts/Roboto-Regular.ttf');
    final ttf = pw.Font.ttf(fontData);
    final image = pw.MemoryImage(_img!);

    // export signature as PNG
    final sigBytes = await _sigController.toPngBytes();
    final sigPm = pw.MemoryImage(sigBytes!);

    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        margin: pw.EdgeInsets.all(32),
        theme: pw.ThemeData(defaultTextStyle: pw.TextStyle(font: ttf)),
        build: (pw.Context context) => [
          pw.Header(
              level: 0,
              text: 'Raport EKG – Rekord ${widget.recordId} ($_lead)'),
          pw.Text('Hasło dostępu: $_reportPassword',
              style: pw.TextStyle(fontSize: 12)),
          pw.SizedBox(height: 10),
          pw.Center(child: pw.Image(image, fit: pw.BoxFit.contain)),
          pw.SizedBox(height: 20),
          pw.Text('Średni QRS: ${_formatMs(_avg!)} ($_qrsClass)'),
          pw.Text('Min QRS: ${_formatMs(_min!)}'),
          pw.Text('Max QRS: ${_formatMs(_max!)}'),
          pw.Text('HR: ${_hr!.toStringAsFixed(1)} bpm ($_hrClass)'),
          pw.SizedBox(height: 20),
          pw.Divider(),
          pw.Header(level: 1, text: 'Notatka lekarza'),
          pw.Paragraph(text: 'Interpretacja: Pacjent bez istotnych odchyleń.'),
          pw.Paragraph(text: 'Zalecenia: Kontrola za 6 miesięcy.'),
          pw.Paragraph(text: 'Podpis elektroniczny:'),
          pw.Center(child: pw.Image(sigPm, width: 150, height: 60)),
        ],
      ),
    );

    final bytes = await pdf.save();
    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/report_${widget.recordId}.pdf');
    await file.writeAsBytes(bytes, flush: true);
    return file;
  }

  Future<void> _sendReport(String method) async {
    if (_img == null || _avg == null) return;
    final subject = 'Raport EKG – rekord ${widget.recordId} ($_lead)';
    final body = 'Rekord: ${widget.recordId}\n'
        'Odprowadzenie: $_lead\n\n'
        'Średni QRS: ${_formatMs(_avg!)} ($_qrsClass)\n'
        'Min QRS: ${_formatMs(_min!)}\n'
        'Max QRS: ${_formatMs(_max!)}\n'
        'HR: ${_hr!.toStringAsFixed(1)} bpm ($_hrClass)\n'
        'Hasło: $_reportPassword';

    try {
      final pdfFile = await _createPdfFile();
      if (method == 'email') {
        final email = Email(
          body: body,
          subject: subject,
          recipients: [],
          attachmentPaths: [pdfFile.path],
          isHTML: false,
        );
        await FlutterEmailSender.send(email);
      } else {
        await Share.shareFiles(
          [pdfFile.path],
          text: body,
          subject: subject,
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Nie udało się udostępnić: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final accent = Theme.of(context).colorScheme.primary;
    return Scaffold(
      appBar: AppBar(
        title: Text('Rekord ${widget.recordId} • $_lead'),
        actions: [
          IconButton(
            icon: const Icon(Icons.multiline_chart),
            tooltip: 'Porównaj wiele odprowadzeń',
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(
                builder: (_) => MultiScreen(recordId: widget.recordId),
              ),
            ),
          ),
          IconButton(
            icon: Icon(_annotationMode ? Icons.edit_off : Icons.edit),
            tooltip: 'Adnotacje',
            onPressed: () => setState(() => _annotationMode = !_annotationMode),
          ),
          DropdownButton<String>(
            value: _lead,
            underline: const SizedBox(),
            icon: const Icon(Icons.show_chart, color: Colors.white),
            items: _leads
                .map((l) => DropdownMenuItem(value: l, child: Text(l)))
                .toList(),
            onChanged: (v) {
              setState(() => _lead = v!);
              _fetchAll();
            },
          ),
          IconButton(
            icon: Icon(_pinned ? Icons.push_pin : Icons.push_pin_outlined),
            tooltip: 'Pokaż/Ukryj hasło do raportu',
            onPressed: () => setState(() => _pinned = !_pinned),
          ),
        ],
      ),
      body: SafeArea(
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : _error != null
                ? Center(
                    child: Text(_error!,
                        style: const TextStyle(color: Colors.red)))
                : SingleChildScrollView(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        if (_pinned)
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: Colors.yellow[100],
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text('Hasło do raportu: $_reportPassword',
                                style: const TextStyle(
                                    fontSize: 16, fontWeight: FontWeight.bold)),
                          ),
                        const SizedBox(height: 8),
                        Container(
                          decoration: BoxDecoration(
                            border: Border.all(color: accent, width: 2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          padding: const EdgeInsets.all(4),
                          child: SizedBox(
                            height: 220,
                            child: Stack(
                              children: [
                                Positioned.fill(
                                  child: InteractiveViewer(
                                    maxScale: 4,
                                    child: Image.memory(_img!,
                                        fit: BoxFit.contain),
                                  ),
                                ),
                                if (_annotationMode) const AnnotationOverlay(),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),
                        const Text('Podpis lekarza:',
                            style: TextStyle(
                                fontSize: 16, fontWeight: FontWeight.bold)),
                        SizedBox(
                          height: 150,
                          child: Signature(
                            controller: _sigController,
                            backgroundColor: Colors.grey[200]!,
                          ),
                        ),
                        TextButton(
                          onPressed: () => _sigController.clear(),
                          child: const Text('Wyczyść podpis'),
                        ),
                        const SizedBox(height: 16),
                        Center(
                          child: Wrap(
                            alignment: WrapAlignment.center,
                            spacing: 12,
                            runSpacing: 12,
                            children: [
                              StatCard(
                                  icon: Icons.av_timer,
                                  label: 'Średni QRS',
                                  value: _formatMs(_avg!)),
                              StatCard(
                                  icon: Icons.timer_off,
                                  label: 'Min QRS',
                                  value: _formatMs(_min!)),
                              StatCard(
                                  icon: Icons.timer,
                                  label: 'Max QRS',
                                  value: _formatMs(_max!)),
                              StatCard(
                                  icon: Icons.favorite,
                                  label: 'HR',
                                  value: '${_hr!.toStringAsFixed(1)} bpm'),
                            ],
                          ),
                        ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              child: ElevatedButton.icon(
                                icon: const Icon(Icons.email),
                                label: const Text('E-mail'),
                                onPressed: () => _sendReport('email'),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ElevatedButton.icon(
                                icon: const Icon(Icons.share),
                                label: const Text('Udostępnij'),
                                onPressed: () => _sendReport('whatsapp'),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
      ),
    );
  }
}
