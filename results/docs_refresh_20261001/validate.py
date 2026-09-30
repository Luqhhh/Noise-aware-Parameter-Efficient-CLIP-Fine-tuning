"""Verify the documentation refresh against its frozen repository baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
BASE = 'f474ff17e6d48b3330d031861d0ba58d7119950c'
LINK = re.compile(r'\[([^\]\n]+)\]\(([^)\n]+)\)')


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', *args], cwd=ROOT)


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding='utf-8')


def slugs(content: str) -> set[str]:
    used: dict[str, int] = {}
    result = set()
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', content, re.M):
        slug = re.sub(r'[^\w\-\s]', '', heading.lower()).replace(' ', '-')
        repeat = used.get(slug, 0)
        used[slug] = repeat + 1
        result.add(slug if repeat == 0 else f'{slug}-{repeat}')
    return result


def verify(include_working_tree: bool) -> dict:
    errors: list[str] = []
    diff_args = ['diff', '--name-only', BASE]
    if not include_working_tree:
        diff_args.append('HEAD')
    changed = set(git(*diff_args).decode().splitlines())
    if include_working_tree:
        changed.update(git('ls-files', '--others', '--exclude-standard').decode().splitlines())
    non_docs = sorted(name for name in changed if not (
        name.endswith('.md') or name.startswith('results/docs_refresh_20261001/')))
    if non_docs:
        errors.append(f'Changes outside documentation/audit scope: {non_docs}')
    readmes = sorted(name for name in git('ls-tree', '-r', '--name-only', BASE).decode().splitlines()
                     if 'readme' in Path(name).name.lower())
    for name in readmes:
        if (ROOT / name).read_bytes() != git('show', f'{BASE}:{name}'):
            errors.append(f'README bytes changed: {name}')

    checked_links = 0
    markdown = sorted(name for name in changed if name.endswith('.md'))
    for name in markdown:
        path = ROOT / name
        if not path.exists():
            errors.append(f'Document removed: {name}')
            continue
        for _, target in LINK.findall(read(name)):
            if target.startswith(('http:', 'https:', '/', 'mailto:')):
                continue
            relative, _, anchor = unquote(target).partition('#')
            destination = (path.parent / relative).resolve() if relative else path
            checked_links += 1
            if not destination.exists():
                errors.append(f'{name}: missing {target}')
            elif anchor and destination.suffix == '.md' and anchor not in slugs(destination.read_text()):
                errors.append(f'{name}: missing anchor {target}')
            elif anchor and re.fullmatch(r'L\d+(?:-L\d+)?', anchor) and destination.is_file():
                last = int(re.findall(r'\d+', anchor)[-1])
                if last > len(destination.read_text().splitlines()):
                    errors.append(f'{name}: invalid source line {target}')

    original = git('show', f'{BASE}:docs/current_execution_plan.md').decode()
    archive = read('docs/history/execution_plan_before_docs_refresh_20261001.md')
    original_digest = hashlib.sha256(original.encode()).hexdigest()
    body = archive.split('\n---\n\n', 1)[1]

    def restore_link(match: re.Match) -> str:
        label, target = match.groups()
        return f'[{label}]({target[3:]})' if target.startswith('../') else match.group(0)

    restored = LINK.sub(restore_link, body)
    if restored != original or original_digest not in archive:
        errors.append('Archived execution entry does not preserve baseline content')

    platform = json.loads(read('results/v1_full_swa_platform_20261001/artifact_verification.json'))
    comparison = json.loads(read('results/v3_final_delivery_20261001/comparison.json'))
    all_rows = comparison['slices']['all']
    budget = platform['target_75_percent']
    current = read('docs/current_execution_plan.md')
    policy = read('docs/p75_error_budget_policy_20260929.md')
    for text in (current, policy):
        for fact in (str(platform['platform_score_percent']), f"{budget['remaining_correct_implied']:,}",
                     platform['artifacts']['zip']['path']):
            # The ZIP path is expressed as a compact CSV/ZIP directory pair in prose.
            needle = fact.rsplit('/', 1)[0] if fact.endswith('/submission.zip') else fact
            if needle not in text:
                errors.append(f'Current status/budget misses source fact: {needle}')
    for fact in ('completed_delivered', comparison['decision'], str(all_rows['corrections']),
                 f"{all_rows['regressions']:,}", f"−{abs(all_rows['net'])}",
                 f"{abs(all_rows['delta_macro_pp']):.4f}",
                 f"{abs(all_rows['delta_micro_pp']):.4f}", platform['artifacts']['zip']['sha256']):
        if fact not in current:
            errors.append(f'Execution entry misses source fact: {fact}')
    if all_rows['corrections'] - all_rows['regressions'] != all_rows['net']:
        errors.append('Paired correction arithmetic differs from archived result')
    if budget['required_correct'] - platform['platform_correct_count_implied'] != budget['remaining_correct_implied']:
        errors.append('Platform target arithmetic differs from archived result')

    # The repaired historical result must retain the four independently recorded scores.
    old_scores = re.findall(r'\*\*(6[12]\.\d+)\*\*', git('show', f'{BASE}:docs/a2_lora_platform_results_2026-07-22.md').decode())
    repaired = read('docs/a2_lora_platform_results_2026-07-22.md')
    for score in old_scores:
        if score + '%' not in repaired:
            errors.append(f'Historical A2 score lost: {score}')

    whitespace = subprocess.run(['git', 'diff', '--check', BASE] + ([] if include_working_tree else ['HEAD']),
                                cwd=ROOT, text=True, capture_output=True)
    if whitespace.returncode:
        errors.append(whitespace.stdout + whitespace.stderr)

    return {
        'record_id': 'DOCS_REFRESH_20261001',
        'base_commit': BASE,
        'status': 'passed' if not errors else 'failed',
        'changed_markdown_files': markdown,
        'changed_markdown_count': len(markdown),
        'readme_files_verified_unchanged': len(readmes),
        'relative_links_checked': checked_links,
        'execution_archive_original_sha256': original_digest,
        'execution_archive_preserved': restored == original,
        'platform_score_percent': platform['platform_score_percent'],
        'remaining_correct_implied_not_receipt': budget['remaining_correct_implied'],
        'v3_net_corrections': all_rows['net'],
        'active_submission': platform['artifacts'],
        'existing_submission_validation': {
            'source': 'results/v1_full_swa_platform_20261001/artifact_verification.json',
            'checks_passed': platform['submission_checks_passed'],
            'rows': platform['submission_rows'],
            'zip_csv_bytes_identical': platform['zip_csv_bytes_identical'],
            'package_revalidated_in_docs_refresh': False,
        },
        'training_started': False,
        'new_candidate_produced': False,
        'errors': errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-working-tree', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.include_working_tree)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered, encoding='utf-8')
    print(json.dumps({key: result[key] for key in (
        'status', 'changed_markdown_count', 'readme_files_verified_unchanged',
        'relative_links_checked', 'execution_archive_preserved', 'errors')}, ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
