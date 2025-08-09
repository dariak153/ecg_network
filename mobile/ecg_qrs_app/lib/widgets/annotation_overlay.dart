import 'package:flutter/material.dart';

class AnnotationOverlay extends StatefulWidget {
  const AnnotationOverlay({Key? key}) : super(key: key);

  @override
  State<AnnotationOverlay> createState() => _AnnotationOverlayState();
}

class _AnnotationOverlayState extends State<AnnotationOverlay> {
  final List<Offset> _points = [];

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapUp: (details) {
        setState(() => _points.add(details.localPosition));
        _showNoteDialog(details.localPosition);
      },
      child: CustomPaint(
        size: Size.infinite,
        painter: _AnnotationPainter(_points),
      ),
    );
  }

  void _showNoteDialog(Offset pos) {
    final ctl = TextEditingController();
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Dodaj notatkę'),
        content: TextField(
          controller: ctl,
          decoration: const InputDecoration(hintText: 'PVC, BBB...'),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Anuluj')),
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Zapisz')),
        ],
      ),
    );
  }
}

class _AnnotationPainter extends CustomPainter {
  final List<Offset> points;
  _AnnotationPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 2;
    for (final pt in points) {
      canvas.drawCircle(pt, 6, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _AnnotationPainter old) =>
      old.points.length != points.length;
}
