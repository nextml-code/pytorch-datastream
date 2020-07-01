#!/usr/bin/env bash
# ./publish.sh 0.13.37
set -o errexit -o nounset -o pipefail -o xtrace

VERSION="$1"
VERSION_TAG="v${VERSION}"

command -v git
command -v python
command -v twine

publish () {
    (
        VERSION="$1"

        python setup.py sdist

        read -r -p "Looks good to publish ${VERSION} to pypi [y/N]? " response
        case "$response" in
            [yY][eE][sS]|[yY])
                twine upload dist/pytorch-datastream-${VERSION}.tar.gz
                ;;
            *)
                echo "Aborted upload"
                exit 5
                ;;
        esac
    )
}

(
    cd "$( dirname "${BASH_SOURCE[0]}" )"

    if [[ $VERSION = v* ]]
    then
        echo "VERSION should be specified without prefixed v"
        exit 6
    fi

    git fetch
    test -z "$(git status --porcelain)" || (echo "Dirty repo"; exit 2)
    test -z "$(git diff origin/master)" || (echo "Not up to date with origin/master"; exit 3)

    git fetch --tags
    git tag -l | sed '/^'"${VERSION_TAG}"'$/{q2}' > /dev/null \
        || (echo "${VERSION_TAG} already exists"; exit 4)

    python -m pytest

    git diff origin/master

    read -r -p "Deploying ${VERSION_TAG}, are you sure? [y/N]? " response
    case "$response" in
        [yY][eE][sS]|[yY])
            git tag "${VERSION_TAG}"
            git push origin "${VERSION_TAG}"
            git push origin master

            publish ${VERSION}
            ;;
        *)
            git checkout .
            echo "Aborted deploy"
            ;;
    esac
)
