
import pygame, sys, os, shutil
from pygame.locals import *
from getopt import getopt

from parser import parse_log_file 

temp_dir = ".spy/"

def usage():
    print "Usage: "+ sys.argv[0] +" file_name"
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        usage()

    opts, args = getopt(sys.argv[1:],'d:h:w:')
    opts = dict(opts)
    if len(args) <> 1: 
        usage()

    height = int(opts.get('-h',640))
    width = int(opts.get('-w',480))
    delta = int(opts.get('-d',20))

    file_name = args[0]
    print 'Loading log file '+file_name+'...'
    log = parse_log_file(file_name)

    pygame.init()
    fpsClock = pygame.time.Clock()

    # Make a temporary directory
    print 'Generating images...'
    tree_images = log.print_trees(temp_dir)
    tree_surfaces = list()
    tree_origins = list()
    for t in tree_images:
        tree_surf = pygame.image.load(t)
        tree_surfaces.append(tree_surf)
        tree_origins.append((0,0))

    print 'Initializing display...'
    depOrigin = (0,0)

    showTree = True # Otherwise show dependence
    currentTree = 0
    surface = pygame.display.set_mode((width,height))
    pygame.display.set_caption('Legion Spy')

    redColor = pygame.Color(255,0,0)
    greenColor = pygame.Color(0,255,0)
    blueColor = pygame.Color(0,0,255)
    whiteColor = pygame.Color(255,255,255)
    blackColor = pygame.Color(0,0,0)

    while True:
        surface.fill(whiteColor)

        if showTree:
            tree_surf = tree_surfaces[currentTree]
            tree_origin = tree_origins[currentTree]
            surface.blit(tree_surf, tree_origin)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == K_h: # move left
                    origin = tree_origins[currentTree]
                    origin = (origin[0]-delta,origin[1])
                    tree_origins[currentTree] = origin
                elif event.key == K_j: # move down
                    origin = tree_origins[currentTree]
                    origin = (origin[0],origin[1]+delta)
                    tree_origins[currentTree] = origin
                elif event.key == K_k: # move up
                    origin = tree_origins[currentTree]
                    origin = (origin[0],origin[1]-delta)
                    tree_origins[currentTree] = origin
                elif event.key == K_l: # move right
                    origin = tree_origins[currentTree]
                    origin = (origin[0]+delta,origin[1])
                    tree_origins[currentTree] = origin
                elif event.key == K_r: # reset
                    tree_origins[currentTree] = (0,0)
                elif event.key == K_LEFT: # next picture 
                    if currentTree == 0:
                        currentTree = len(tree_images)-1
                    else:
                        currentTree = currentTree-1
                elif event.key == K_RIGHT: # previous picture
                    currentTree = ((currentTree+1) % len(tree_images))

        pygame.display.update()
        fpsClock.tick(30)

if __name__ == "__main__":
    try:
        os.mkdir(temp_dir)
        main()
        shutil.rmtree(temp_dir)
    except:
        # Remove the directory we created
        shutil.rmtree(temp_dir)
        raise

