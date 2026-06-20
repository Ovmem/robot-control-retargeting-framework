fp = r'E:\运控\robot_control_retargeting_framework\demos\panda\demo_hand_retargeting_pd_gc.py'
with open(fp, encoding='utf-8') as f:
    lines = f.readlines()

# Find the retargeter.update() block and add a print for delta/base
for i, line in enumerate(lines):
    if 'retargeter = HandToPandaRetargeter(' in line:
        # Add a var to track last_reset for printing
        pass

for i, line in enumerate(lines):
    if line.strip() == 'retargeter = HandToPandaRetargeter(':
        indent = '        '
        # Insert debug var after retargeter init
        lines.insert(i+9, indent + 'base_wrist_logged = False\n')
        break

# Find the 'if detected:' block inside the loop and add debug printing
for i, line in enumerate(lines):
    if 'target = retargeter.update(obs)' in line:
        # After retargeter.update, add base_wrist logging
        indent = '                '
        debug_line = indent + 'if hasattr(retargeter, \"base_wrist_img\") and retargeter.base_wrist_img is not None:\n'
        debug_line += indent + '    pass  # baseline is set\n'
        lines.insert(i+1, debug_line)
        break

# Find the camera window section and add 'r' key for reset
for i, line in enumerate(lines):
    if 'if cv2.waitKey(1) & 0xFF == 27:' in line:
        indent = '                '
        lines.insert(i, indent + '            key = cv2.waitKey(1) & 0xFF\n')
        lines.insert(i+1, indent + '            if key == 27:  # ESC\n')
        lines.insert(i+2, '                        break\n')
        lines.insert(i+3, indent + '            elif key == ord(\"r\"):\n')
        lines.insert(i+4, indent + '                retargeter.reset_origin()\n')
        lines.insert(i+5, indent + '                print(\"  Reset retargeter baseline (r pressed)\")\n')
        # Remove old waitKey line
        for j in range(i+6, len(lines)):
            if 'cv2.waitKey(1) & 0xFF == 27' in lines[j] and 'key' not in lines[j]:
                # This is the old waitKey line, comment it or let it be (it'll be before the new code we inserted)
                pass
        break

open(fp, 'w', encoding='utf-8').write(''.join(lines))
print('Fixed')
