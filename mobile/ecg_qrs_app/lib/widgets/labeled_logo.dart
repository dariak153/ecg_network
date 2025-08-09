import 'package:flutter/material.dart';

class LabeledLogo extends StatelessWidget {
  final bool isEnglish;
  final double size;

  const LabeledLogo({Key? key, required this.isEnglish, this.size = 100})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Image.asset(
          'assets/logo.png',
          width: size,
          height: size,
        ),
        const SizedBox(height: 8),
        Text(
          isEnglish ? 'QRS Detector' : 'Detektor QRS',
          style: TextStyle(
            fontSize: size * 0.2,
            fontWeight: FontWeight.bold,
            color: Theme.of(context).colorScheme.primary,
          ),
        ),
      ],
    );
  }
}
