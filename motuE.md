name: PR Maintainer Checklist

on:
  pull_request_target:
    branches:
      - main
    types:
      - opened

jobs:
  add_pr_checklist:
    runs-on: ubuntu-latest
    name: Add PR Maintainer Checklist
    permissions:
      pull-requests: write
    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Get PR author's pull request count
        id: pr_count
        uses: actions/github-script@v6
        with:
          github-token: ${{secrets.GITHUB_TOKEN}}
          script: |
            const { data: pullRequests } = await github.rest.pulls.list({
              owner: context.repo.owner,
              repo: context.repo.repo,
              state: 'all',
              creator: context.payload.pull_request.user.login
            });
            return pullRequests.length;

      - name: Comment PR
        uses: thollander/actions-comment-pull-request@v2
        with:
          message: |
            ## Thank you for the pull request!
            
            This is PR number ${{ steps.pr_count.outputs.result }} from @${{ github.event.pull_request.user.login }} in this repository.
            ${{ steps.pr_count.outputs.result == '1' && '🎉 Congratulations on your first pull request in this repository! 🎉' || '' }}
            
            ### Maintainer Checklist:
            - [ ] Review the changes
            - [ ] Check for code quality and style
            - [ ] Ensure tests pass
            - [ ] Update documentation if needed
            - [ ] Approve or request changes
