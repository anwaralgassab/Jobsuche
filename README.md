  ...
  - name: Commit changes (seen_jobs.json + drafts)
    run: |
      git config --local user.email "github-actions[bot]@users.noreply.github.com"
      git config --local user.name "github-actions[bot]"
      git add seen_jobs.json || true
      git add drafts || true
      if git diff --staged --quiet; then
        echo "No changes to commit"
      else
        git pull --rebase origin ${{ github.ref_name }} || true
        git commit -m "chore(job-agent): update seen_jobs and drafts [skip ci]" || true
        git push origin HEAD || (git push origin HEAD:job-agent/update-seen && gh pr create --base main --head job-agent/update-seen --title "chore(job-agent): update seen_jobs" --body "Auto-update seen_jobs/drafts")
      fi
