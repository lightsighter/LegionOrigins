
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

    ctx_images = log.print_contexts(temp_dir)
    ctx_surfaces = list()
    ctx_origins = list()
    ctx_ids = list()
    for ctx_id,c in ctx_images.iteritems():
        ctx_surf = pygame.image.load(c)
        ctx_surfaces.append(ctx_surf)
        ctx_origins.append((0,0))
        ctx_ids.append(ctx_id)

    print 'Initializing display...'

    showTree = True # Otherwise show dependence
    currentTree = 0
    currentCtx = 0
    surface = pygame.display.set_mode((width,height))
    pygame.display.set_caption('Legion Spy')

    redColor = pygame.Color(255,0,0)
    greenColor = pygame.Color(0,255,0)
    blueColor = pygame.Color(0,0,255)
    whiteColor = pygame.Color(255,255,255)
    blackColor = pygame.Color(0,0,0)

    fontObj = pygame.font.Font(pygame.font.get_default_font(),32)

    while True:
        surface.fill(whiteColor)

        if showTree:
            tree_surf = tree_surfaces[currentTree]
            tree_origin = tree_origins[currentTree]
            surface.blit(tree_surf, tree_origin)
        else:
            ctx_surf = ctx_surfaces[currentCtx]
            ctx_origin = ctx_origins[currentCtx]
            title = 'Context '+str(ctx_ids[currentCtx])
            msgSurfaceObj = fontObj.render(title,False,blackColor)
            msgRect = msgSurfaceObj.get_rect()
            surface.blit(ctx_surf, (ctx_origin[0],ctx_origin[1]+msgRect.height))
            surface.blit(msgSurfaceObj, (0,0))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == K_h: # move left
                    if showTree:
                        origin = tree_origins[currentTree]
                        origin = (origin[0]-delta,origin[1])
                        tree_origins[currentTree] = origin
                    else:
                        origin = ctx_origins[currentCtx]
                        origin = (origin[0]-delta,origin[1])
                        ctx_origins[currentCtx] = origin
                elif event.key == K_j: # move down
                    if showTree:
                        origin = tree_origins[currentTree]
                        origin = (origin[0],origin[1]+delta)
                        tree_origins[currentTree] = origin
                    else:
                        origin = ctx_origins[currentCtx]
                        origin = (origin[0],origin[1]+delta)
                        ctx_origins[currentCtx] = origin
                elif event.key == K_k: # move up
                    if showTree:
                        origin = tree_origins[currentTree]
                        origin = (origin[0],origin[1]-delta)
                        tree_origins[currentTree] = origin
                    else:
                        origin = ctx_origins[currentCtx]
                        origin = (origin[0],origin[1]-delta)
                        ctx_origins[currentCtx] = origin
                elif event.key == K_l: # move right
                    if showTree:
                        origin = tree_origins[currentTree]
                        origin = (origin[0]+delta,origin[1])
                        tree_origins[currentTree] = origin
                    else:
                        origin = ctx_origins[currentCtx]
                        origin = (origin[0]+delta,origin[1])
                        ctx_origins[currentCtx] = origin
                elif event.key == K_r: # reset
                    if showTree:
                        tree_origins[currentTree] = (0,0)
                    else:
                        ctx_origins[currentCtx] = (0,0)
                elif event.key == K_LEFT: # next picture 
                    if showTree:
                        if currentTree == 0:
                            currentTree = len(tree_images)-1
                        else:
                            currentTree = currentTree-1
                    else:
                        if currentCtx == 0:
                            currentCtx = len(ctx_images)-1
                        else:
                            currentCtx = currentCtx-1
                elif event.key == K_RIGHT: # previous picture
                    if showTree:
                        currentTree = ((currentTree+1) % len(tree_images))
                    else:
                        currentCtx = ((currentCtx+1) % len(ctx_images))
                elif (event.key == K_UP) or (event.key == K_DOWN):
                    showTree = not(showTree)

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

